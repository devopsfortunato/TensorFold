//! ``tensorfold-native``: ``capabilities --json`` for the Python switch, and ``serve MODEL [flags]`` with no Python.
const std = @import("std");
const json = @import("json.zig");
const cli = @import("cli.zig");
const log = @import("log.zig");
const hub = @import("hub.zig");
const engines = @import("engines.zig");
const serve = @import("serve.zig");
const hf_text = @import("hf_text.zig");
const vision = @import("vision.zig");

const usage_line = "usage: tensorfold serve [-h] [--host HOST] [--port PORT] [--name NAME] [--alias ALIAS] [--api-key API_KEY] [--api-key-file API_KEY_FILE] [--metrics-open] [--dashboard] [--context CONTEXT] [--speed-up SETTINGS] [--prompt-cache-gib PROMPT_CACHE_GIB] [--prompt-cache-over-cap] [--max-tokens MAX_TOKENS] [--temperature TEMPERATURE] [--top-p TOP_P] [--top-k TOP_K] [--min-p MIN_P] [--thinking | --no-thinking] [--reasoning-effort {low,medium,high,xhigh}] [--thinking-budget THINKING_BUDGET] [--loop-guard] [--no-drafts] [--parallel PARALLEL] [--no-update-check] [--backend {auto,mlx,cuda}] model\n";

pub fn main(init: std.process.Init) !u8 {
    const gpa = init.gpa;
    const io = init.io;
    const a = init.arena.allocator();
    const argv = try init.minimal.args.toSlice(a);
    log.init(io, false);
    if (argv.len == 3 and std.mem.eql(u8, argv[1], "capabilities") and std.mem.eql(u8, argv[2], "--json")) {
        var out: std.Io.Writer.Allocating = .init(a);
        try cli.capabilities(&out.writer, engines.capabilities(a));
        try std.Io.File.stdout().writeStreamingAll(io, out.written());
        return 0;
    }
    if (argv.len < 2 or !std.mem.eql(u8, argv[1], "serve")) {
        std.debug.print("usage: tensorfold-native capabilities --json | serve MODEL [flags]\n", .{});
        return 2;
    }
    const started = std.Io.Clock.awake.now(io).toNanoseconds();
    var u: cli.Usage = .{};
    const args = cli.parse(a, argv[2..], &u) catch |e| switch (e) {
        error.Usage => {
            std.debug.print("{s}tensorfold serve: error: {s}\n", .{ usage_line, u.message });
            return 2;
        },
        else => |x| return x,
    };
    var problem: []const u8 = "";
    const dir = try hub.resolve(a, io, init.environ_map, args.model, &problem) orelse return fail(problem);
    const text = hf_text.HfText.load(gpa, io, dir, &problem) catch return fail(problem);
    defer text.deinit();
    const model_type = modelType(a, io, dir);
    // --vision (rank 0): the helper loads and warms the tower on this GPU before the engine reads its memory
    // budget, so the tower and the helper's context stay outside the caches' share
    var helper: ?*vision.Helper = null;
    defer if (helper) |h| h.stop(io);
    if (args.vision and args.rank == 0) {
        const env_int = struct {
            fn of(m: *const std.process.Environ.Map, name: []const u8, default: u32) u32 {
                const t = m.get(name) orelse return default;
                return std.fmt.parseInt(u32, std.mem.trim(u8, t, " "), 10) catch default;
            }
        };
        const settings: vision.Settings = .{
            .allow_urls = args.vision_urls,
            .max_images = if (args.vision_max_images) |n| @intCast(n) else env_int.of(init.environ_map, "TENSORFOLD_MAX_IMAGES", 50),
            .max_videos = if (args.vision_max_videos) |n| @intCast(n) else env_int.of(init.environ_map, "TENSORFOLD_MAX_VIDEOS", 4),
            .image_tokens = if (args.vision_image_tokens) |n| @intCast(n) else env_int.of(init.environ_map, "TENSORFOLD_IMAGE_TOKENS", 16384),
        };
        helper = vision.Helper.start(gpa, io, dir, settings, init.environ_map, &problem) catch return fail(problem);
    }
    const opened = try engines.open(a, gpa, io, dir, model_type, args, &problem) orelse return fail(problem);
    defer opened.close(opened.ctx);
    if (args.rank != 0) {
        // a following rank serves no API: it runs rank 0's requests in step until rank 0 stops
        const follow = opened.follow orelse return fail("this engine has no follower for --rank 1");
        follow(opened.ctx) catch |e| return fail(try std.fmt.allocPrint(a, "rank {d} stopped: {s}", .{ args.rank, @errorName(e) }));
        return 0;
    }
    return serve.run(gpa, io, args, .{
        .engine = opened.engine,
        .text = text.text(),
        .served = hub.servedName(args.name, args.model, dir),
        .sampling = try sampling(a, io, dir, args),
        .environ = init.environ_map,
        .started = started,
        .vision = helper,
    });
}

fn fail(message: []const u8) u8 {
    std.debug.print("tensorfold: {s}\n", .{message});
    return 1;
}

fn modelType(a: std.mem.Allocator, io: std.Io, dir: []const u8) []const u8 {
    const path = std.fs.path.join(a, &.{ dir, "config.json" }) catch return "unknown";
    const bytes = std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(16 << 20)) catch return "unknown";
    const doc = std.json.parseFromSliceLeaky(std.json.Value, a, bytes, .{}) catch return "unknown";
    if (doc != .object) return "unknown";
    const t = doc.object.get("model_type") orelse return "unknown";
    return if (t == .string) t.string else "unknown";
}

/// generation_config.json's sampling (``do_sample`` false is greedy), then the serve flags over it.
fn sampling(a: std.mem.Allocator, io: std.Io, dir: []const u8, args: cli.Args) !?json.Value {
    const out = try json.newObject(a);
    const path = try std.fs.path.join(a, &.{ dir, "generation_config.json" });
    if (std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 20))) |bytes| {
        if ((try json.parse(a, bytes)) == .ok) {
            const cfg = (try json.parse(a, bytes)).ok;
            for ([_][]const u8{ "temperature", "top_k", "top_p", "min_p" }) |k| if (cfg.field(k)) |v| try out.put(a, k, v);
            if (cfg.get("do_sample")) |d| if (d == .bool) {
                if (!d.bool) try out.put(a, "temperature", .{ .float = 0 }) else if (out.get("temperature") == null) try out.put(a, "temperature", .{ .float = 1 });
            };
        }
    } else |_| {}
    if (args.temperature) |t| try out.put(a, "temperature", .{ .float = t });
    if (args.top_p) |t| try out.put(a, "top_p", .{ .float = t });
    if (args.top_k) |t| try out.put(a, "top_k", try json.intValue(a, t));
    if (args.min_p) |t| try out.put(a, "min_p", .{ .float = t });
    return if (out.count() > 0) json.Value{ .object = out } else null;
}
