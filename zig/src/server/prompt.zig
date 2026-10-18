//! Prompt ids from a request, as ``prepare_prompt`` and ``render_prompt_ids`` make them.
const std = @import("std");
const json = @import("json.zig");
const errors = @import("errors.zig");
const messages_mod = @import("messages.zig");
const model_text = @import("model_text.zig");
const chat = @import("chat.zig");
const Server = @import("server.zig").Server;
const api = @import("engine_api");
const vision = @import("vision.zig");
const log = @import("log.zig");
const Value = json.Value;
const Cx = errors.Cx;

pub const Rendered = struct { ids: []const u32, history_len: usize = 0, media: ?*const api.Media = null };

pub const isTitle = messages_mod.isTitleRequest;

/// ``render_prompt_ids``: messages normalized for this template, rendered, and a dangling ``<think>`` closed.
pub fn renderIds(srv: *Server, cx: *Cx, messages: Value, tools: []const Value, thinking: bool, effort: ?[]const u8, generation: bool) errors.Refused![]const u32 {
    const normalized = try messages_mod.toolArguments(cx, try messages_mod.normalize(cx, messages, srv.late_system, srv.needs_user_after_tool));
    var problem: []const u8 = "";
    const options: model_text.RenderOptions = .{
        .tools = if (tools.len > 0) Value{ .array = @constCast(tools) } else null,
        .add_generation_prompt = generation,
        .enable_thinking = thinking,
        .reasoning_effort = if (thinking) effort else null,
    };
    var ids = srv.text.renderIds(cx.a, normalized, options, &problem) catch |e| switch (e) {
        error.OutOfMemory => return error.OutOfMemory,
        error.Template => return cx.fail(.server, "{s}", .{problem}),
    };
    if (!thinking and generation and ids.len > 0) {
        const last = try srv.text.decode(cx.a, ids[ids.len - 1 ..]);
        if (std.mem.eql(u8, @import("reply_text.zig").pyStrip(last), "<think>")) {
            const close = srv.text.encode(cx.a, "</think>", false) catch |e| switch (e) {
                error.OutOfMemory => return error.OutOfMemory,
                error.Template => &[_]u32{},
            };
            if (close.len == 1) ids = try std.mem.concat(cx.a, u32, &.{ ids, close });
        }
    }
    return ids;
}

/// A request's prompt: raw text or ids for a completion, else the chat template's, with its history length.
pub fn prepare(srv: *Server, cx: *Cx, input: chat.Input, thinking: bool, effort: ?[]const u8) errors.Refused!Rendered {
    if (input.prompt) |p| switch (p) {
        .text => |t| return .{ .ids = srv.text.encode(cx.a, t, true) catch |e| switch (e) {
            error.OutOfMemory => return error.OutOfMemory,
            error.Template => return cx.fail(.server, "the tokenizer cannot encode this prompt", .{}),
        } },
        .ids => |ids| return .{ .ids = ids },
    };
    if (input.media) |raw| return prepareMedia(srv, cx, raw, input.tools, thinking, effort);
    const prompt = try renderIds(srv, cx, input.messages, input.tools, thinking, effort, true);
    const history = try renderIds(srv, cx, input.messages, input.tools, thinking, effort, false);
    var history_len: usize = 0;
    if (history.len > 0 and history.len < prompt.len and std.mem.eql(u32, prompt[0..history.len], history)) history_len = history.len;
    if (history_len == 0 and prompt.len > 1 and std.mem.eql(u32, history, prompt)) history_len = prompt.len - 1; // a template with no generation suffix
    return .{ .ids = prompt, .history_len = history_len };
}

/// ``prepare_images``: the messages split (image_url / video_url parts as the template's image / video parts, the
/// media in order), normalized with their parts, rendered, then the vision helper's processor and tower: Python's
/// token ids and the media the engine takes. Such a prompt is neither kept nor resumed.
fn prepareMedia(srv: *Server, cx: *Cx, raw: Value, tools: []const Value, thinking: bool, effort: ?[]const u8) errors.Refused!Rendered {
    const helper = srv.vision orelse return cx.refuse("image input requires a supported vision checkpoint served with --vision");
    if (!srv.info.media) return cx.refuse("image input requires a supported vision checkpoint served with --vision");
    var sources: std.ArrayList(vision.Source) = .empty;
    const template = try messages_mod.splitMedia(cx, raw, srv.media_limits, &sources);
    if (!helper.videos) for (sources.items) |src| if (src.kind == .video) return cx.refuse("content parts must be text or image_url; audio and video are unsupported");
    const normalized = try messages_mod.toolArguments(cx, try messages_mod.normalizeWith(cx, template, srv.late_system, srv.needs_user_after_tool, true));
    var problem: []const u8 = "";
    const options: model_text.RenderOptions = .{
        .tools = if (tools.len > 0) Value{ .array = @constCast(tools) } else null,
        .add_generation_prompt = true,
        .enable_thinking = thinking,
        .reasoning_effort = if (thinking) effort else null,
    };
    const text = srv.text.render(cx.a, normalized, options, &problem) catch |e| switch (e) {
        error.OutOfMemory => return error.OutOfMemory,
        error.Template => return cx.fail(.request, "the chat template rejected the request: {s}", .{problem}),
    };
    const window = srv.info.context_window;
    const result = helper.prepare(cx.a, text, sources.items, if (window > 0) window else null) catch |e| switch (e) {
        error.OutOfMemory => return error.OutOfMemory,
        else => return cx.fail(.capacity, "image processing failed ({s}); retry shortly", .{@errorName(e)}),
    };
    switch (result) {
        .refused => |r| {
            if (r.context) return cx.fail(.context_length, "{s}", .{r.message});
            return cx.fail(if (r.status == 503) .capacity else .request, "{s}", .{r.message});
        },
        .ok => |p| {
            log.line("vision: {d} image(s), {d} video(s): {d} visual tokens of {d} prompt tokens (prepare {d:.3}s, tower {d:.3}s, helper GPU peak {d:.2} GiB, features {s})", .{ p.images, p.videos, p.visual_tokens, p.tokens.len, p.prepare_s, p.encode_s, @as(f64, @floatFromInt(p.peak)) / (1 << 30), if (p.features_sha.len >= 16) p.features_sha[0..16] else p.features_sha });
            return .{ .ids = p.tokens, .media = p.media };
        },
    }
}

/// A reusable system prefix, found with a probe in place of the first user message; zero below 512 tokens.
pub fn systemPrefixLen(srv: *Server, cx: *Cx, messages: Value, tools: []const Value, prompt_ids: []const u32, thinking: bool, effort: ?[]const u8) usize {
    if (messages != .array) return 0;
    const first_user = for (messages.array, 0..) |m, i| {
        const role = m.get("role") orelse continue;
        if (role == .string and std.mem.eql(u8, role.string, "user")) break i;
    } else return 0;
    const probe = cx.a.alloc(Value, first_user + 1) catch return 0;
    @memcpy(probe[0..first_user], messages.array[0..first_user]);
    const user = json.newObject(cx.a) catch return 0;
    user.put(cx.a, "role", .{ .string = "user" }) catch return 0;
    user.put(cx.a, "content", .{ .string = "\u{2063}probe" }) catch return 0;
    probe[first_user] = .{ .object = user };
    var scratch: Cx = .{ .a = cx.a };
    const other = renderIds(srv, &scratch, .{ .array = probe }, tools, thinking, effort, true) catch return 0;
    var shared: usize = 0;
    while (shared < @min(prompt_ids.len, other.len) and prompt_ids[shared] == other[shared]) shared += 1;
    return if (shared >= 512) shared else 0;
}
