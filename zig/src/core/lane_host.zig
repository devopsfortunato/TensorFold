//! The lane core served to the HTTP threads: one thread owns ``core`` and steps rounds while any stream lives.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const pc = @import("prompt_cache.zig");
const Allocator = std.mem.Allocator;
const Id = api.Id;
const Request = api.Request;
const Sink = api.Sink;
const Event = api.Event;
const Engine = api.Engine;
const Info = api.Info;
const Status = api.Status;
const Memory = api.Memory;
const Reason = api.Reason;
const Stats = api.Stats;
const SubmitError = api.SubmitError;

pub const LaneHost = struct {
    gpa: Allocator,
    io: std.Io,
    core: *lanes.Engine,
    info_: Info,
    min_match: i64 = 4,
    mutex: std.Io.Mutex = .init,
    wake: std.Io.Condition = .init,
    queued: std.ArrayList(*Job) = .empty,
    admitted: std.ArrayList(*Job) = .empty,
    cancels: std.ArrayList(Id) = .empty,
    closing: bool = false,
    thread: ?std.Thread = null,
    decoded: std.ArrayList(Mark) = .empty, // tokens a round landed, for the 2 s decode rate
    prefill_rate: f64 = 0,
    prefill_at: i96 = 0,
    live_tokens: std.ArrayList(u32) = .empty,
    live_generated: u64 = 0,
    lone: ?api.Lone = null, // the backend's driver for a lone greedy stream; null: every stream in the lane core
    lone_job: ?*Job = null, // the job that driver holds now
    cache: ?*pc.Store = null, // kept prompt states (engine thread only); the backend restores and saves them
    round_log: bool = false, // TF_FLASHNEXT_ROUND_LOG: the engine thread's time by part, every 200 steps
    t_step: i96 = 0,
    t_deliver: i96 = 0,
    t_steps: u64 = 0,
    admit_wait_ms: u32 = 0, // an idle engine waits this long after a first arrival for requests sent together (a burst prefills in one pass)

    const Mark = struct { at: i96, tokens: u64 };
    const window_ns: i96 = 2 * std.time.ns_per_s;

    const Job = struct {
        host: *LaneHost,
        id: Id,
        request: *const Request,
        sink: Sink,
        stream: lanes.Stream = undefined,
        proposer: lanes.SuffixLookup = undefined,
        delivered: usize = 0,
        started: bool = false,
        prefill_sent: bool = false, // a lone driver's prefilled event went out
        began: i96 = 0,
        prefilled: ?i96 = null,
        entry: ?*pc.Entry = null, // the kept state the backend restores, until its prompt pass reports
        entry_id: u64 = 0, // its id: a keep while this stream's begin waits (fair fills) may evict it
        source: ?u64 = null, // that state's id, once its restore went through: a keep of this pass may supersede it
        marks: []const u32 = &.{}, // where the pass keeps states (gpa-owned)
        kept0: u64 = 0, // the store's kept count when the job looked it up
        key: []const u32 = &.{}, // what the store matches: the prompt, or a media prompt's `match`

        /// The backend's prompt pass stands at a mark: the cache keeps the stream's state there.
        fn kept(ptr: *anyopaque, s: *lanes.Stream, at: u32) void {
            const job: *Job = @ptrCast(@alignCast(ptr));
            job.reported(); // before a keep can evict the entry the pass restored
            if (job.host.cache) |store| _ = store.keepFrom(job.key, at, s, job.source);
        }

        /// The pass started from its kept state, or its copy failed (the entry goes); a pass that never got there: neither.
        fn reported(job: *Job) void {
            const e = job.entry orelse return;
            job.entry = null;
            const store = job.host.cache orelse return;
            if (!job.started) return;
            job.stream.reuse.saved = null; // the backend restores before its first chunk; a later keep may free it
            if (!store.holds(e, job.entry_id)) return; // evicted before the pass began (it prefilled from 0): nothing to note
            if (job.stream.reuse_failed) store.resumed(e, job.key, false) else if (job.stream.cached == e.at) {
                job.source = e.id;
                store.resumed(e, job.key, true);
            }
        }
    };

    pub fn init(gpa: Allocator, io: std.Io, core: *lanes.Engine, info_: Info) LaneHost {
        var enforced = info_;
        enforced.loop_guard = true;
        return .{ .gpa = gpa, .io = io, .core = core, .info_ = enforced };
    }

    /// The lane core's word that a filling prompt is in (or its fill ended): emit `prefilled`, or end the job.
    fn filled(ptr: *anyopaque, s: *lanes.Stream, err: ?anyerror) void {
        const h: *LaneHost = @ptrCast(@alignCast(ptr));
        const job: *Job = @alignCast(@fieldParentPtr("stream", s));
        const e = err orelse return h.prefilled(job, job.began); // the run loop delivers its tokens
        if (e == error.Cancelled) {
            _ = h.cancel(job);
        } else _ = h.drop(job, @errorName(e));
    }

    pub fn start(h: *LaneHost) !void {
        h.core.fill_hook = .{ .ptr = h, .done = filled };
        h.thread = try std.Thread.spawn(.{ .stack_size = 16 << 20 }, run, .{h});
    }

    /// Stops admitting, cancels what is left and joins the engine thread.
    pub fn stop(h: *LaneHost) void {
        h.mutex.lockUncancelable(h.io);
        h.closing = true;
        h.wake.broadcast(h.io);
        h.mutex.unlock(h.io);
        if (h.thread) |t| t.join();
        h.thread = null;
        h.queued.deinit(h.gpa);
        h.admitted.deinit(h.gpa);
        h.cancels.deinit(h.gpa);
        h.decoded.deinit(h.gpa);
        h.live_tokens.deinit(h.gpa);
    }

    pub fn engine(h: *LaneHost) Engine {
        return .{ .ctx = h, .vtable = &.{ .info = infoFn, .submit = submitFn, .cancel = cancelFn, .status = statusFn, .memory = memoryFn } };
    }

    fn self(ctx: *anyopaque) *LaneHost {
        return @ptrCast(@alignCast(ctx));
    }

    fn infoFn(ctx: *anyopaque) Info {
        return self(ctx).info_;
    }

    fn submitFn(ctx: *anyopaque, id: Id, request: *const Request, sink: Sink) SubmitError!void {
        const h = self(ctx);
        const job = h.gpa.create(Job) catch return error.Busy;
        job.* = .{ .host = h, .id = id, .request = request, .sink = sink };
        h.mutex.lockUncancelable(h.io);
        defer h.mutex.unlock(h.io);
        if (h.closing) {
            h.gpa.destroy(job);
            return error.Closed;
        }
        // foreground before background, each in arrival order (the Python job queue's priority)
        var at = h.queued.items.len;
        if (!request.background) {
            while (at > 0 and h.queued.items[at - 1].request.background) at -= 1;
        }
        h.queued.insert(h.gpa, at, job) catch {
            h.gpa.destroy(job);
            return error.Busy;
        };
        h.wake.signal(h.io);
    }

    fn cancelFn(ctx: *anyopaque, id: Id) void {
        const h = self(ctx);
        h.mutex.lockUncancelable(h.io);
        defer h.mutex.unlock(h.io);
        h.cancels.append(h.gpa, id) catch {};
        h.wake.signal(h.io);
    }

    fn statusFn(ctx: *anyopaque, out: *Status, stream_tokens: []u32) void {
        const h = self(ctx);
        h.mutex.lockUncancelable(h.io);
        defer h.mutex.unlock(h.io);
        const now = std.Io.Clock.awake.now(h.io).toNanoseconds();
        var tokens: u64 = 0;
        for (h.decoded.items) |m| {
            if (m.at >= now - window_ns) tokens += m.tokens;
        }
        const n = @min(stream_tokens.len, h.live_tokens.items.len);
        @memcpy(stream_tokens[0..n], h.live_tokens.items[0..n]);
        out.* = .{
            .running = @intCast(h.admitted.items.len),
            .waiting = @intCast(h.queued.items.len),
            .decode_tokens_per_second = @as(f64, @floatFromInt(tokens)) / 2.0,
            .prefill_tokens_per_second = if (now - h.prefill_at <= window_ns) h.prefill_rate else 0,
            .preemptions = 0,
            .streams = n,
            .generation_tokens = h.live_generated,
        };
    }

    fn memoryFn(_: *anyopaque, _: bool) ?Memory {
        return null;
    }

    fn emit(job: *Job, event: Event) void {
        job.sink.event(job.sink.ctx, job.id, &event);
    }

    /// A job's cancel hook for its prompt pass: its id is in `cancels`, or the host is closing (read under the lock).
    fn cancelled(ctx: *anyopaque) bool {
        const job: *Job = @ptrCast(@alignCast(ctx));
        job.host.lock();
        defer job.host.unlock();
        return job.host.closing or std.mem.indexOfScalar(Id, job.host.cancels.items, job.id) != null;
    }

    /// Hands a stream the tokens its rounds committed since the last delivery.
    fn send(h: *LaneHost, job: *Job) void {
        const emitted = job.stream.emitted();
        if (emitted.len > job.delivered) {
            emit(job, .{ .tokens = emitted[job.delivered..] });
            h.noteDecoded(emitted.len - job.delivered);
            job.delivered = emitted.len;
        }
    }

    /// Sends a stream's new tokens; true once it has finished (its job freed).
    fn deliver(h: *LaneHost, job: *Job) bool {
        h.send(job);
        if (!job.stream.finished) return false;
        const reason: Reason = switch (job.stream.reason) {
            .length => .length,
            .cancelled => .cancelled,
            .@"error" => .failed,
            else => .stop,
        };
        h.finish(job, reason, job.stream.problem);
        return true;
    }

    fn finish(h: *LaneHost, job: *Job, reason: Reason, message: []const u8) void {
        job.reported();
        h.gpa.free(job.marks);
        const s = &job.stream;
        const stats: Stats = if (job.started) .{ .rounds = s.rounds, .drafted = s.drafted, .accepted = s.accepted, .min_rows = s.min_rows, .loop_period = s.loop_period, .prefill_seconds = if (job.prefilled) |done| @as(f64, @floatFromInt(@as(i64, @intCast(@max(0, done - job.began))))) / 1e9 else null } else .{};
        emit(job, .{ .finished = .{ .reason = reason, .stats = stats, .message = message } });
        if (job.started) {
            if (s.grammar) |g| g.deinit(h.gpa);
            s.deinit(h.gpa);
            job.proposer.deinit();
        }
        h.gpa.destroy(job);
    }

    fn noteDecoded(h: *LaneHost, n: usize) void {
        const now = std.Io.Clock.awake.now(h.io).toNanoseconds();
        h.mutex.lockUncancelable(h.io);
        defer h.mutex.unlock(h.io);
        var keep: usize = 0;
        for (h.decoded.items) |m| {
            if (m.at < now - window_ns) continue;
            h.decoded.items[keep] = m;
            keep += 1;
        }
        h.decoded.shrinkRetainingCapacity(keep);
        h.decoded.append(h.gpa, .{ .at = now, .tokens = n }) catch {};
    }

    /// Cancels queued and admitted jobs named since the last round.
    fn takeCancels(h: *LaneHost) void {
        h.mutex.lockUncancelable(h.io);
        const ids = h.gpa.dupe(Id, h.cancels.items) catch &.{};
        h.cancels.clearRetainingCapacity();
        var dropped: std.ArrayList(*Job) = .empty;
        for (ids) |id| {
            for (h.queued.items, 0..) |job, i| if (job.id == id) {
                dropped.append(h.gpa, h.queued.orderedRemove(i)) catch {};
                break;
            };
        }
        h.mutex.unlock(h.io);
        for (dropped.items) |job| h.finish(job, .cancelled, "");
        dropped.deinit(h.gpa);
        for (ids) |id| {
            for (h.admitted.items, 0..) |job, i| if (job.id == id) {
                if (!job.stream.finished) h.core.discard(&job.stream);
                h.lock();
                _ = h.admitted.orderedRemove(i);
                h.unlock();
                h.finish(job, .cancelled, "");
                break;
            };
        }
        h.gpa.free(ids);
    }

    fn lock(h: *LaneHost) void {
        h.mutex.lockUncancelable(h.io);
    }

    fn unlock(h: *LaneHost) void {
        h.mutex.unlock(h.io);
    }

    const Prep = union(enum) { none, dropped, job: *Job };

    /// Takes the next queued request into a free lane and builds its stream, not yet prefilled: `none` when none waits
    /// or no lane is free, `dropped` when it ended here.
    fn admitPrepare(h: *LaneHost) Prep {
        h.lock();
        if (h.queued.items.len == 0 or h.admitted.items.len >= h.info_.lanes) {
            h.unlock();
            return .none;
        }
        const job = h.queued.orderedRemove(0);
        h.admitted.append(h.gpa, job) catch {
            h.unlock();
            h.finish(job, .failed, "out of memory");
            return .dropped;
        };
        h.unlock();
        const r = job.request;
        var reuse: lanes.stream.Reuse = .{};
        // nothing keeps between here and the backend's begin, so `saved` is live there; a fair fill's begin can then
        // wait while other prompts' keeps (or admission reclaims) free the entry: the backend resumes only the state it
        // saw at its begin (cuda_lanes Filling.gen), else prefills from 0, and `reported` notes an entry only while the
        // store holds it
        // a prompt with images or video is matched by its `match` ids (each image's content hash in its placeholder
        // rows): without them, or on an engine that cannot resume one, it is neither resumed nor kept
        job.key = if (r.media == null) r.prompt else if (h.info_.media_reuse and r.match.len == r.prompt.len) r.match else &.{};
        if (job.key.len > 0) if (h.cache) |store| if (store.lookup(h.gpa, job.key, r.history_len, r.shared_prefixes, r.chunks)) |l| {
            job.entry = l.entry;
            job.entry_id = if (l.entry) |e| e.id else 0;
            job.kept0 = store.counts.kept;
            job.marks = l.marks;
            reuse = .{ .saved = if (l.entry) |e| e.saved else null, .at = if (l.entry) |e| e.at else 0, .marks = l.marks, .hook = .{ .ptr = job, .at = Job.kept } };
        } else |_| {};
        job.proposer = lanes.SuffixLookup.init(h.gpa, .{ .min_match = h.min_match }) catch return h.droppedJob(job, "the drafter could not start");
        const grammar: ?*lanes.grammar.Constraint = if (r.structure) |st| h.constraint(st) catch |e| {
            job.proposer.deinit();
            return h.droppedJob(job, if (e == error.OutOfMemory) "out of memory" else "the reply's grammar could not start (xgrammar)");
        } else null;
        job.stream = lanes.Stream.init(h.gpa, .{
            .id = "request",
            .prompt = r.prompt,
            .max_new = r.max_tokens,
            .eos = r.eos,
            .sampling = r.sampling,
            .drafts = r.drafts,
            .proposer = job.proposer.proposer(),
            .stop_check = if (r.stop) |s| .{ .ptr = s.ctx, .check = s.check } else null,
            .cancel_check = .{ .ptr = job, .check = cancelled },
            .think_budget = r.think_budget,
            .think_close = r.think_close,
            .think_end = if (r.think_end) |t| t else -1,
            .loop_guard = r.loop_guard,
            .call = if (r.call) |c| .{
                .opener = c.opener,
                .think_open = if (c.think_open) |t| t else -1,
                .think_end = if (c.think_end) |t| t else -1,
                .lead = c.lead,
                .names = c.names,
                .tail = c.tail,
                .lex = .{ .ctx = c.lex.ctx, .blank = c.lex.blank, .text = c.lex.text, .encode = c.lex.encode },
            } else null,
            .chunks = r.chunks,
            .reuse = reuse,
            .media = r.media,
            .grammar = grammar,
        }) catch {
            if (grammar) |g| g.deinit(h.gpa);
            job.proposer.deinit();
            return h.droppedJob(job, "out of memory");
        };
        job.started = true;
        job.began = std.Io.Clock.awake.now(h.io).toNanoseconds();
        return .{ .job = job };
    }

    fn droppedJob(h: *LaneHost, job: *Job, message: []const u8) Prep {
        _ = h.drop(job, message);
        return .dropped;
    }

    /// A reply's grammar from the one the server compiled (Info.structures engines get requests that carry one): a
    /// matcher only, so the engine thread never waits for a compile.
    fn constraint(h: *LaneHost, st: api.Structure) !*lanes.grammar.Constraint {
        const g = st.compiled orelse return error.NoCompiledGrammar;
        return g.constraint(h.gpa, st.after);
    }

    /// Every request waiting for a lane, admitted now: prompts that arrive together (a burst, with no resume or kept
    /// state to take) prefill in one backend call, where the backend can (one pass over their rows).
    fn admitAll(h: *LaneHost) void {
        var burst: [16]*Job = undefined;
        var n: usize = 0;
        while (true) switch (h.admitPrepare()) {
            .none => break,
            .dropped => {},
            .job => |job| {
                if (h.loneFits(job)) {
                    _ = h.runLone(job, job.began);
                } else if (n < burst.len and h.core.batches() and job.stream.media == null and job.stream.grammar == null and job.stream.reuse.at == 0 and job.stream.reuse.marks.len == 0) {
                    burst[n] = job;
                    n += 1;
                } else _ = h.startOne(job);
            },
        };
        h.startBurst(burst[0..n]);
    }

    /// The jobs' prompts in one backend call, then each joined and delivered in admission order; a refused burst
    /// prefills one at a time.
    fn startBurst(h: *LaneHost, jobs: []*Job) void {
        if (jobs.len == 0) return;
        if (jobs.len == 1) {
            _ = h.startOne(jobs[0]);
            return;
        }
        var ss: [16]*lanes.Stream = undefined;
        const began = std.Io.Clock.awake.now(h.io).toNanoseconds();
        for (jobs, ss[0..jobs.len]) |job, *s| {
            job.began = began;
            s.* = &job.stream;
        }
        h.core.prefillBurst(ss[0..jobs.len]) catch |e| {
            for (jobs) |job| {
                if (e == error.BurstRefused) _ = h.startOne(job) else if (e == error.Cancelled) _ = h.cancel(job) else _ = h.drop(job, @errorName(e));
            }
            return;
        };
        for (jobs) |job| {
            h.core.addPrefilled(&job.stream) catch |e| {
                _ = if (e == error.Cancelled) h.cancel(job) else h.drop(job, @errorName(e));
                continue;
            };
            h.prefilled(job, began);
            if (h.deliver(job)) h.remove(job);
        }
    }

    /// One prepared job: its prompt prefilled alone (or its fill started), delivered.
    fn startOne(h: *LaneHost, job: *Job) bool {
        const began = job.began;
        // a backend that fills prompts between rounds starts this one and admits the next (prompts that arrive
        // together fill together); the fill hook reports it in
        const filling = h.core.beginStream(&job.stream) catch |e| return if (e == error.Cancelled) h.cancel(job) else h.drop(job, @errorName(e));
        if (filling) return true;
        h.prefilled(job, began);
        if (h.deliver(job)) h.remove(job);
        return true;
    }

    fn prefilled(h: *LaneHost, job: *Job, began: i96) void {
        const done = std.Io.Clock.awake.now(h.io).toNanoseconds();
        h.lock();
        if (done > began) h.prefill_rate = @as(f64, @floatFromInt(job.request.prompt.len - job.stream.cached)) / (@as(f64, @floatFromInt(done - began)) / 1e9);
        h.prefill_at = done;
        job.prefilled = done;
        h.unlock();
        job.reported();
        if (h.cache) |store| {
            store.report(job.request.prompt.len, job.stream.cached, store.counts.kept - job.kept0);
            if (job.key.len > 0) store.noteServed(job.key); // text: the prompt; an image prompt: its match ids (media reuse on)
        }
        emit(job, .{ .prefilled = job.stream.cached });
    }

    /// A greedy drafted request alone in the engine, with nothing waiting: the backend's own driver takes it.
    fn loneFits(h: *LaneHost, job: *Job) bool {
        const r = job.request;
        if (h.lone == null or r.sampling != null or !r.drafts or r.think_budget > 0 or r.loop_guard or r.call != null or r.structure != null or r.media != null) return false;
        h.lock();
        defer h.unlock();
        return h.admitted.items.len == 1 and h.queued.items.len == 0 and h.cancels.items.len == 0 and h.core.activeCount() == 0 and h.core.fillingCount() == 0;
    }

    /// The lone driver's rounds, its tokens sent as they land; a request arriving or a cancel hands the stream to the lane core.
    fn runLone(h: *LaneHost, job: *Job, began: i96) bool {
        h.lone_job = job;
        job.delivered = 0;
        const lone = h.lone.?;
        const Hooks = struct {
            fn committed(ctx: *anyopaque) void {
                const host: *LaneHost = @ptrCast(@alignCast(ctx));
                const j = host.lone_job.?;
                if (!j.prefill_sent) {
                    j.prefill_sent = true;
                    host.prefilled(j, j.began);
                }
                host.send(j);
                host.noteLive();
            }
            fn yield(ctx: *anyopaque) bool {
                const host: *LaneHost = @ptrCast(@alignCast(ctx));
                host.lock();
                defer host.unlock();
                return host.queued.items.len > 0 or host.cancels.items.len > 0 or host.closing;
            }
        };
        job.began = began;
        const paused = lone.run(lone.ctx, &job.stream, .{ .ctx = h, .committed = Hooks.committed, .yield = Hooks.yield });
        h.lone_job = null;
        const handed = paused catch |e| {
            if (!job.prefill_sent) emit(job, .{ .prefilled = 0 });
            h.remove(job);
            h.finish(job, if (e == error.Cancelled) .cancelled else .failed, if (e == error.Cancelled) "" else @errorName(e));
            return true;
        };
        if (!job.prefill_sent) h.prefilled(job, began);
        if (handed) {
            h.core.adopt(&job.stream) catch |e| return h.drop(job, @errorName(e));
            h.send(job);
            return true;
        }
        if (h.deliver(job)) h.remove(job);
        return true;
    }

    fn drop(h: *LaneHost, job: *Job, message: []const u8) bool {
        h.remove(job);
        if (job.started and !job.stream.finished) h.core.discard(&job.stream);
        h.finish(job, .failed, message);
        return true;
    }

    /// A job cancelled in its prompt pass, its lane already released.
    fn cancel(h: *LaneHost, job: *Job) bool {
        h.remove(job);
        h.finish(job, .cancelled, "");
        return true;
    }

    fn remove(h: *LaneHost, job: *Job) void {
        h.lock();
        defer h.unlock();
        for (h.admitted.items, 0..) |j, i| if (j == job) {
            _ = h.admitted.orderedRemove(i);
            return;
        };
    }

    fn noteLive(h: *LaneHost) void {
        h.lock();
        defer h.unlock();
        h.live_tokens.clearRetainingCapacity();
        h.live_generated = 0;
        for (h.admitted.items) |job| if (job.started) {
            h.live_tokens.append(h.gpa, @intCast(job.stream.context.items.len)) catch {};
            h.live_generated += @intCast(job.stream.emitted().len);
        };
    }

    /// The engine is idle and a request has just arrived: requests sent together reach the queue a few ms apart, so wait
    /// briefly (`admit_wait_ms`) and admit them as one burst instead of one prompt pass each.
    fn gatherArrivals(h: *LaneHost) void {
        if (h.admit_wait_ms == 0) return;
        h.lock();
        const lone = h.queued.items.len > 0 and h.queued.items.len < h.info_.lanes and h.admitted.items.len == 0 and h.cancels.items.len == 0 and !h.closing;
        h.unlock();
        if (!lone or h.core.activeCount() != 0 or h.core.fillingCount() != 0) return;
        std.Io.sleep(h.io, .fromMilliseconds(h.admit_wait_ms), .awake) catch {};
    }

    fn run(h: *LaneHost) void {
        while (true) {
            h.takeCancels();
            h.gatherArrivals();
            h.admitAll();
            h.noteLive();
            h.lock();
            if (h.closing) {
                const left = h.queued.items.len + h.admitted.items.len;
                h.unlock();
                if (left == 0) return;
                h.closeAll();
                continue;
            }
            if (h.core.activeCount() == 0 and h.core.fillingCount() == 0) {
                if (h.cancels.items.len == 0 and (h.queued.items.len == 0 or h.admitted.items.len >= h.info_.lanes))
                    h.wake.waitTimeout(h.io, &h.mutex, .{ .duration = .{ .raw = .fromMilliseconds(100), .clock = .awake } }) catch {};
                h.unlock();
                continue;
            }
            h.unlock();
            const t0: i96 = if (h.round_log) std.Io.Clock.awake.now(h.io).toNanoseconds() else 0;
            h.core.step() catch |e| {
                h.failAll(@errorName(e));
                continue;
            };
            const t1: i96 = if (h.round_log) std.Io.Clock.awake.now(h.io).toNanoseconds() else 0;
            defer if (h.round_log) {
                h.t_step += t1 - t0;
                h.t_deliver += std.Io.Clock.awake.now(h.io).toNanoseconds() - t1;
                h.t_steps += 1;
                if (h.t_steps % 200 == 0) {
                    const n: f64 = 200;
                    std.log.info("round log: {d:.2} ms a step in the lane core, {d:.2} ms delivering tokens", .{ @as(f64, @floatFromInt(h.t_step)) / 1e6 / n, @as(f64, @floatFromInt(h.t_deliver)) / 1e6 / n });
                    h.t_step = 0;
                    h.t_deliver = 0;
                }
            };
            var i: usize = 0;
            while (i < h.admitted.items.len) {
                const job = h.admitted.items[i];
                if (h.deliver(job)) {
                    h.lock();
                    _ = h.admitted.orderedRemove(i);
                    h.unlock();
                } else i += 1;
            }
        }
    }

    /// A failed round ends every stream it held, with the backend's error.
    fn failAll(h: *LaneHost, message: []const u8) void {
        h.lock();
        const jobs = h.gpa.dupe(*Job, h.admitted.items) catch &.{};
        h.admitted.clearRetainingCapacity();
        h.unlock();
        for (jobs) |job| {
            if (!job.stream.finished) h.core.discard(&job.stream);
            h.finish(job, .failed, message);
        }
        h.gpa.free(jobs);
    }

    fn closeAll(h: *LaneHost) void {
        h.lock();
        for (h.queued.items) |job| h.cancels.append(h.gpa, job.id) catch {};
        for (h.admitted.items) |job| h.cancels.append(h.gpa, job.id) catch {};
        h.unlock();
        h.takeCancels();
    }
};

test "a lane host serves the core's own tokens, in order, and cancels between rounds" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
    try host.start();
    defer host.stop();
    const Box = struct {
        mutex: std.Io.Mutex = .init,
        tokens: std.ArrayList(u32) = .empty,
        done: ?Reason = null,
        fn event(ctx: *anyopaque, _: Id, e: *const Event) void {
            const b: *@This() = @ptrCast(@alignCast(ctx));
            b.mutex.lockUncancelable(std.testing.io);
            defer b.mutex.unlock(std.testing.io);
            switch (e.*) {
                .tokens => |t| b.tokens.appendSlice(gpa, t) catch {},
                .finished => |f| b.done = f.reason,
                else => {},
            }
        }
        fn wait(b: *@This()) Reason {
            while (true) {
                b.mutex.lockUncancelable(std.testing.io);
                const d = b.done;
                b.mutex.unlock(std.testing.io);
                if (d) |r| return r;
                std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
            }
        }
    };
    const prompt = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6 };
    var box: Box = .{};
    defer box.tokens.deinit(gpa);
    const request: Request = .{ .prompt = &prompt, .max_tokens = 24 };
    const e = host.engine();
    try e.submit(1, &request, .{ .ctx = &box, .event = Box.event });
    try std.testing.expectEqual(Reason.length, box.wait());
    var history: std.ArrayList(u32) = .empty;
    defer history.deinit(gpa);
    try history.appendSlice(gpa, &prompt);
    for (box.tokens.items) |t| {
        try std.testing.expectEqual(lanes.fake.next(history.items, null, history.items.len), t);
        try history.append(gpa, t);
    }
    try std.testing.expectEqual(@as(usize, 24), box.tokens.items.len);
    var gone: Box = .{};
    defer gone.tokens.deinit(gpa);
    const long: Request = .{ .prompt = &prompt, .max_tokens = 100000 };
    try e.submit(2, &long, .{ .ctx = &gone, .event = Box.event });
    e.cancel(2);
    try std.testing.expectEqual(Reason.cancelled, gone.wait());

    const CancelPrefill = struct {
        engine: Engine,
        id: Id,
        at: usize,

        fn call(ctx: *anyopaque, _: *lanes.Stream, chunk: usize) void {
            const c: *@This() = @ptrCast(@alignCast(ctx));
            if (chunk == c.at) c.engine.cancel(c.id);
        }
    };
    const chunked_prompt = [_]u32{ 8, 6, 7, 5, 3, 0, 9, 2, 1, 4 };
    var chunked: Box = .{};
    defer chunked.tokens.deinit(gpa);
    var prefill_cancel = CancelPrefill{ .engine = e, .id = 3, .at = 2 };
    target.prefill_chunks = 10;
    target.prefill_count = 0;
    target.prefill_hook = CancelPrefill.call;
    target.prefill_hook_ctx = &prefill_cancel;
    const chunked_request: Request = .{ .prompt = &chunked_prompt, .max_tokens = 1 };
    try e.submit(3, &chunked_request, .{ .ctx = &chunked, .event = Box.event });
    try std.testing.expectEqual(Reason.cancelled, chunked.wait());
    try std.testing.expect(target.prefill_count <= 3);
    try std.testing.expectEqual(@as(usize, 0), target.lanes.count()); // its lane released

    // the lone driver's prompt pass (gpu_round.run starts with Backend.opening), cancelled the same way
    const Lone = struct {
        be: lanes.backend.Backend,

        fn run(ctx: *anyopaque, s: *lanes.Stream, _: api.LoneHooks) anyerror!bool {
            const l: *@This() = @ptrCast(@alignCast(ctx));
            _ = try l.be.opening(gpa, s);
            return error.NotCancelled;
        }
    };
    var lone: Box = .{};
    defer lone.tokens.deinit(gpa);
    var lone_driver = Lone{ .be = target.backend() };
    host.lone = .{ .ctx = &lone_driver, .run = Lone.run };
    prefill_cancel.id = 4;
    target.prefill_count = 0;
    try e.submit(4, &chunked_request, .{ .ctx = &lone, .event = Box.event });
    try std.testing.expectEqual(Reason.cancelled, lone.wait());
    try std.testing.expect(target.prefill_count <= 3);
    try std.testing.expectEqual(@as(usize, 0), target.lanes.count());
}
