//! Chat messages as the template sees them: leading instructions merged, text parts joined, call arguments as objects.
const std = @import("std");
const json = @import("json.zig");
const errors = @import("errors.zig");
const fields = @import("fields.zig");
const Value = json.Value;
const Cx = errors.Cx;

const roles = [_][]const u8{ "system", "developer", "user", "assistant", "tool" };

fn isRole(role: ?Value) bool {
    const r = role orelse return false;
    if (r != .string) return false;
    for (roles) |name| if (std.mem.eql(u8, name, r.string)) return true;
    return false;
}

fn withField(cx: *Cx, o: *const json.Object, key: []const u8, value: Value) !*json.Object {
    const copy = try json.copyObject(cx.a, o);
    try copy.put(cx.a, key, value);
    return copy;
}

/// ``has_images``: a message whose content holds an image_url or video_url part.
pub fn hasVisual(messages: ?Value) bool {
    const list = messages orelse return false;
    if (list != .array) return false;
    for (list.array) |m| {
        const content = m.get("content") orelse continue;
        if (content != .array) continue;
        for (content.array) |part| if (part.typeIs("image_url") or part.typeIs("video_url")) return true;
    }
    return false;
}

const media_keys = [_][]const u8{ "image", "images", "image_url", "input_image", "audio", "input_audio", "video", "video_url" };

fn mediaBesides(part: Value, keep: []const u8) bool {
    for (media_keys) |k| if (!std.mem.eql(u8, k, keep) and json.truthyField(part, k)) return true;
    return false;
}

/// Media limits ``split_images`` checks: Flash Next's (the single-Spark recipe's patches 0008 and 0009).
pub const MediaLimits = struct {
    max_images: u32 = 50,
    max_videos: u32 = 4,
    image_bytes: usize = 10 * 1024 * 1024,
    video_bytes: usize = 64 * 1024 * 1024,
    max_url_chars: usize = 4096,
    allow_urls: bool = false,
};

/// A request's messages for the template and its media in order (``split_images`` with videos on): each
/// image_url part (user and tool messages) becomes ``{"type": "image", "detail": d}``, each video_url part (user
/// messages) ``{"type": "video"}``; text parts stay as they are. Python's checks and messages.
pub fn splitMedia(cx: *Cx, messages: ?Value, limits: MediaLimits, sources: *std.ArrayList(@import("vision.zig").Source)) errors.Refused!Value {
    const list = messages orelse return cx.refuse("messages must be a non-empty list");
    if (list != .array or list.array.len == 0) return cx.refuse("messages must be a non-empty list");
    var out: std.ArrayList(Value) = .empty;
    var images: u32 = 0;
    var videos: u32 = 0;
    for (list.array) |message| {
        if (message != .object) return cx.refuse("each message must be an object");
        const role = message.get("role");
        if (!isRole(role)) return cx.refuse("invalid message role");
        for (media_keys) |k| if (json.truthyField(message, k)) return cx.refuse("images must be image_url parts in user or tool message content");
        const content = message.get("content");
        if (content == null or content.? == .null or content.? == .string) {
            try out.append(cx.a, message);
            continue;
        }
        if (content.? != .array) return cx.refuse("message content must be text or an array of content parts");
        var parts: std.ArrayList(Value) = .empty;
        for (content.?.array) |part| {
            if (part != .object) return cx.refuse("each content part must be an object");
            if (part.typeIs("text")) {
                const t = part.get("text");
                if (t == null or t.? != .string or mediaBesides(part, "")) return cx.refuse("text parts must contain a text string without media");
                try parts.append(cx.a, part);
            } else if (part.typeIs("video_url")) {
                if (!std.mem.eql(u8, role.?.string, "user")) return cx.refuse("video_url parts are supported only in user messages");
                if (mediaBesides(part, "video_url")) return cx.refuse("video_url parts cannot contain other media");
                if (videos >= limits.max_videos) return cx.fail(.request, "a request supports at most {d} videos; send fewer, or restart the server with --vision-max-videos N to raise the count limit", .{limits.max_videos});
                const v = part.get("video_url");
                const url = if (v) |x| x.strField("url") else null;
                if (url == null or url.?.len == 0) return cx.refuse("video_url must contain a non-empty url string");
                if (std.mem.startsWith(u8, url.?, "data:")) {
                    if (url.?.len > limits.video_bytes * 3 / 2 + 256) return cx.refuse("video data URL exceeds the encoded byte limit");
                } else if (!std.mem.startsWith(u8, url.?, "https://")) {
                    return cx.refuse("videos require data URLs or public HTTPS URLs");
                } else if (!limits.allow_urls) {
                    return cx.refuse("video URLs are off on this server; send the video as a data URL, or start the server with --vision-urls");
                } else if (url.?.len > limits.max_url_chars) return cx.refuse("video URL is too long");
                try sources.append(cx.a, .{ .kind = .video, .url = url.? });
                videos += 1;
                const o = try json.newObject(cx.a);
                try o.put(cx.a, "type", .{ .string = "video" });
                try parts.append(cx.a, .{ .object = o });
            } else if (part.typeIs("image_url")) {
                const r = role.?.string;
                if (!std.mem.eql(u8, r, "user") and !std.mem.eql(u8, r, "tool")) return cx.refuse("image_url parts are supported only in user and tool messages");
                if (mediaBesides(part, "image_url")) return cx.refuse("image_url parts cannot contain other media");
                if (images >= limits.max_images) return cx.fail(.request, "a request supports at most {d} images across the full message history, including prior turns; remove older image content, start a new conversation, or restart the server with --vision-max-images N to raise the count limit (other image limits still apply)", .{limits.max_images});
                const v = part.get("image_url");
                const url = if (v) |x| x.strField("url") else null;
                if (v == null or v.? != .object or url == null or url.?.len == 0) return cx.refuse("image_url must contain a non-empty url string");
                var detail: []const u8 = "auto";
                if (v.?.get("detail")) |d| {
                    const ok = d == .string and (std.mem.eql(u8, d.string, "auto") or std.mem.eql(u8, d.string, "low") or std.mem.eql(u8, d.string, "high"));
                    if (!ok) return cx.refuse("image detail must be auto, low or high");
                    detail = d.string;
                }
                if (std.mem.startsWith(u8, url.?, "data:")) {
                    if (url.?.len > limits.image_bytes * 3 + 256) return cx.refuse("image data URL exceeds the encoded byte limit");
                } else if (!std.mem.startsWith(u8, url.?, "https://")) {
                    return cx.refuse("images require data URLs or public HTTPS URLs");
                } else if (!limits.allow_urls) {
                    return cx.refuse("image URLs are off on this server; send the image as a data URL, or start the server with --vision-urls");
                } else if (url.?.len > limits.max_url_chars) return cx.refuse("image URL is too long");
                try sources.append(cx.a, .{ .kind = .image, .url = url.?, .detail = detail });
                images += 1;
                const o = try json.newObject(cx.a);
                try o.put(cx.a, "type", .{ .string = "image" });
                try o.put(cx.a, "detail", .{ .string = detail });
                try parts.append(cx.a, .{ .object = o });
            } else return cx.refuse("content parts must be text, image_url or video_url; audio is unsupported");
        }
        try out.append(cx.a, .{ .object = try withField(cx, message.object, "content", .{ .array = parts.items }) });
    }
    return .{ .array = out.items };
}

fn isVisualPart(part: Value) bool {
    return part.typeIs("image_url") or part.typeIs("image") or part.typeIs("video_url") or part.typeIs("video");
}

/// ``normalize_messages`` (text only): leading system and developer text merged, later ones as ``late_system``.
/// A template that needs a user query gains one user turn after a trailing tool run (``needs_user_after_tool``).
pub fn normalize(cx: *Cx, messages: ?Value, late_system: []const u8, needs_user_after_tool: bool) errors.Refused!Value {
    return normalizeWith(cx, messages, late_system, needs_user_after_tool, false);
}

/// ``normalize_messages``; `allow_media` (``allow_images``): a user or tool message with image or video parts
/// keeps its parts list for the template.
pub fn normalizeWith(cx: *Cx, messages: ?Value, late_system: []const u8, needs_user_after_tool: bool, allow_media: bool) errors.Refused!Value {
    const list = messages orelse return cx.refuse("messages must be a non-empty list");
    if (list != .array or list.array.len == 0) return cx.refuse("messages must be a non-empty list");
    var out: std.ArrayList(Value) = .empty;
    var instructions: std.ArrayList(*json.Object) = .empty;
    for (list.array) |message| {
        if (message != .object) return cx.refuse("each message must be an object");
        const role = message.get("role");
        if (!isRole(role)) return cx.refuse("message role must be system, developer, user, assistant or tool");
        if (fields.hasMedia(message)) return cx.refuse("this server accepts text only; image, audio and video inputs are unsupported");
        const content = message.get("content");
        var item: *json.Object = message.object;
        const visual = allow_media and content != null and content.? == .array and for (content.?.array) |part| {
            if (part == .object and isVisualPart(part)) break true;
        } else false;
        if (visual) {
            const r = role.?.string;
            if (!std.mem.eql(u8, r, "user") and !std.mem.eql(u8, r, "tool")) return cx.refuse("images and videos are supported only in user and tool messages");
            for (content.?.array) |part| {
                if (part != .object or !(part.typeIs("text") or isVisualPart(part))) return cx.refuse("image messages may contain text and image_url (or video_url) parts only");
                if (part.typeIs("text")) {
                    const t = part.get("text");
                    if (t == null or t.? != .string) return cx.refuse("a text content part must contain a text string");
                }
            }
            try out.append(cx.a, message);
            continue;
        }
        if (content != null and content.? == .array) {
            var text: std.ArrayList(u8) = .empty;
            for (content.?.array) |part| {
                const typed = part == .object and part.get("type") != null and part.get("type").? == .string and std.mem.eql(u8, part.get("type").?.string, "text");
                if (!typed or fields.hasMedia(part)) return cx.refuse("this server accepts text parts only; image, audio and video inputs are unsupported");
                const t = part.get("text") orelse return cx.refuse("a text content part must contain a text string");
                if (t != .string) return cx.refuse("a text content part must contain a text string");
                try text.appendSlice(cx.a, t.string);
            }
            item = try withField(cx, message.object, "content", .{ .string = text.items });
        } else if (content == null or content.? == .null) {
            item = try withField(cx, message.object, "content", .{ .string = "" });
        } else if (content.? != .string) {
            return cx.refuse("message content must be text or an array of text parts");
        }
        const r = role.?.string;
        if (std.mem.eql(u8, r, "system") or std.mem.eql(u8, r, "developer")) {
            if (out.items.len == 0) {
                try instructions.append(cx.a, item);
                continue;
            }
            if (!std.mem.eql(u8, r, late_system)) item = try withField(cx, item, "role", .{ .string = late_system });
        }
        try out.append(cx.a, .{ .object = item });
    }
    if (instructions.items.len > 0) {
        var joined: std.ArrayList(u8) = .empty;
        for (instructions.items, 0..) |m, i| {
            if (i > 0) try joined.appendSlice(cx.a, "\n\n");
            try joined.appendSlice(cx.a, m.get("content").?.string);
        }
        const first = try withField(cx, instructions.items[0], "role", .{ .string = "system" });
        try first.put(cx.a, "content", .{ .string = joined.items });
        try out.insert(cx.a, 0, .{ .object = first });
    }
    // A template that demands a user query (it raises "No user query found" when none exists) refuses a
    // conversation whose user turn became tool results. By the template's own test a query is a user message
    // whose trimmed text is not wholly a "<tool_response>...</tool_response>" block; only a conversation with
    // none gains one placeholder user turn after its last tool run, the tool messages themselves unchanged.
    if (needs_user_after_tool) {
        var has_query = false;
        var last_tool: ?usize = null;
        for (out.items, 0..) |m, i| {
            const role = m.get("role");
            if (role == null or role.? != .string) continue;
            if (std.mem.eql(u8, role.?.string, "user")) {
                const content = m.get("content");
                const text = if (content != null and content.? == .string) content.?.string else "";
                const trimmed = std.mem.trim(u8, text, " \t\n\r\x0b\x0c");
                if (!(std.mem.startsWith(u8, trimmed, "<tool_response>") and std.mem.endsWith(u8, trimmed, "</tool_response>"))) has_query = true;
            } else if (std.mem.eql(u8, role.?.string, "tool")) {
                last_tool = i;
            }
        }
        if (!has_query) {
            const at = if (last_tool) |i| i + 1 else out.items.len;
            const turn = try json.newObject(cx.a);
            try turn.put(cx.a, "role", .{ .string = "user" });
            try turn.put(cx.a, "content", .{ .string = "(tool results above)" });
            try out.insert(cx.a, at, .{ .object = turn });
        }
    }
    return .{ .array = out.items };
}

/// ``_normalize_tool_call_arguments``: call arguments as objects for templates; bad ones under ``_invalid_arguments``.
pub fn toolArguments(cx: *Cx, messages: Value) errors.Refused!Value {
    if (messages != .array) return messages;
    const out = try cx.a.alloc(Value, messages.array.len);
    for (messages.array, out) |message, *slot| {
        slot.* = message;
        const calls = message.get("tool_calls") orelse continue;
        if (calls != .array or calls.array.len == 0) continue;
        var touched = false;
        const new_calls = try cx.a.alloc(Value, calls.array.len);
        for (calls.array, new_calls) |call, *dst| {
            dst.* = call;
            const function = call.get("function") orelse continue;
            if (function != .object or !function.has("arguments")) continue;
            const args = function.get("arguments").?;
            if (args == .object) continue;
            var parsed: ?Value = args;
            if (args == .string) {
                parsed = switch (try json.parseText(cx.a, args.string)) {
                    .ok => |v| v,
                    .err => null,
                };
            }
            if (parsed == null or parsed.? != .object) {
                const wrapped = try json.newObject(cx.a);
                try wrapped.put(cx.a, "_invalid_arguments", args);
                parsed = .{ .object = wrapped };
            }
            const fn_copy = try withField(cx, function.object, "arguments", parsed.?);
            dst.* = .{ .object = try withField(cx, call.object, "function", .{ .object = fn_copy }) };
            touched = true;
        }
        if (touched) slot.* = .{ .object = try withField(cx, message.object, "tool_calls", .{ .array = new_calls }) };
    }
    return .{ .array = out };
}

/// ``{"type": "text", "text": text}``: a chat content part.
pub fn textPart(a: std.mem.Allocator, text: []const u8) std.mem.Allocator.Error!Value {
    const p = try json.newObject(a);
    try p.put(a, "type", .{ .string = "text" });
    try p.put(a, "text", .{ .string = text });
    return .{ .object = p };
}

/// A short title request without tools (``is_title_request``): it yields to foreground turns.
pub fn isTitleRequest(messages: Value, has_tools: bool) bool {
    if (has_tools or messages != .array or messages.array.len == 0) return false;
    const first = messages.array[0];
    const role = first.get("role") orelse return false;
    if (role != .string or !std.mem.eql(u8, role.string, "system")) return false;
    var size: usize = 0;
    for (messages.array) |m| {
        const c = m.get("content");
        size += if (c != null and c.? == .string) std.unicode.utf8CountCodepoints(c.?.string) catch c.?.string.len else 4096;
    }
    const text = first.get("content") orelse return false;
    if (size >= 4096 or text != .string) return false;
    return std.ascii.findIgnoreCase(text.string, "title") != null;
}

test "a conversation with a user query stays byte-identical under the rule" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    var cx: Cx = .{ .a = arena.allocator() };
    const messages = (try json.parse(cx.a,
        \\[{"role": "user", "content": "run it"}, {"role": "assistant", "content": "", "tool_calls": [{"id": "t1", "type": "function", "function": {"name": "f", "arguments": {}}}]}, {"role": "tool", "tool_call_id": "t1", "content": "42 rows"}, {"role": "assistant", "content": "", "tool_calls": [{"id": "t2", "type": "function", "function": {"name": "g", "arguments": {}}}]}, {"role": "tool", "tool_call_id": "t2", "content": "ok"}, {"role": "assistant", "content": "done"}]
    )).ok;
    const out = try normalize(&cx, messages, "system", true);
    // byte-identical: the same messages, in order, nothing gained
    try std.testing.expectEqualStrings(try json.stringify(cx.a, messages, .{ .ascii = false }), try json.stringify(cx.a, out, .{ .ascii = false }));
}

test "a conversation with no user query gains one placeholder turn after its last tool run" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    var cx: Cx = .{ .a = arena.allocator() };
    const messages = (try json.parse(cx.a,
        \\[{"role": "assistant", "content": "", "tool_calls": [{"id": "t1", "type": "function", "function": {"name": "f", "arguments": {}}}]}, {"role": "tool", "tool_call_id": "t1", "content": "42 rows"}, {"role": "tool", "tool_call_id": "t2", "content": "ok"}]
    )).ok;
    const out = try normalize(&cx, messages, "system", true);
    try std.testing.expectEqual(@as(usize, 4), out.array.len);
    try std.testing.expectEqualStrings("tool", out.array[1].get("role").?.string);
    try std.testing.expectEqualStrings("42 rows", out.array[1].get("content").?.string); // the tool messages unchanged
    try std.testing.expectEqualStrings("tool", out.array[2].get("role").?.string);
    try std.testing.expectEqualStrings("ok", out.array[2].get("content").?.string);
    try std.testing.expectEqualStrings("user", out.array[3].get("role").?.string);
    try std.testing.expectEqualStrings("(tool results above)", out.array[3].get("content").?.string);
}

test "a user message whose content is wholly a tool block does not count as a query" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    var cx: Cx = .{ .a = arena.allocator() };
    const user = try json.newObject(cx.a);
    try user.put(cx.a, "role", .{ .string = "user" });
    try user.put(cx.a, "content", .{ .string = "<tool_response>\ngot 42 rows\n</tool_response>" });
    const assistant = try json.newObject(cx.a);
    try assistant.put(cx.a, "role", .{ .string = "assistant" });
    try assistant.put(cx.a, "content", .{ .string = "" });
    const tool = try json.newObject(cx.a);
    try tool.put(cx.a, "role", .{ .string = "tool" });
    try tool.put(cx.a, "tool_call_id", .{ .string = "t1" });
    try tool.put(cx.a, "content", .{ .string = "done" });
    const messages: Value = .{ .array = try cx.a.dupe(Value, &.{ .{ .object = user }, .{ .object = assistant }, .{ .object = tool } }) };
    const out = try normalize(&cx, messages, "system", true);
    try std.testing.expectEqual(@as(usize, 4), out.array.len);
    try std.testing.expectEqualStrings("user", out.array[3].get("role").?.string);
    try std.testing.expectEqualStrings("(tool results above)", out.array[3].get("content").?.string);
}

test "a conversation keeps its shape when the template does not need the rule" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    var cx: Cx = .{ .a = arena.allocator() };
    const messages = (try json.parse(cx.a,
        \\[{"role": "assistant", "content": "", "tool_calls": [{"id": "t1", "type": "function", "function": {"name": "f", "arguments": {}}}]}, {"role": "tool", "tool_call_id": "t1", "content": "42 rows"}]
    )).ok;
    const out = try normalize(&cx, messages, "system", false);
    try std.testing.expectEqual(@as(usize, 2), out.array.len);
    try std.testing.expectEqualStrings("assistant", out.array[0].get("role").?.string);
    try std.testing.expectEqualStrings("tool", out.array[1].get("role").?.string);
}

test "split_images: image and video parts become the template's parts, the media in order" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    var cx: Cx = .{ .a = arena.allocator() };
    const messages = (try json.parse(cx.a,
        \\[{"role": "system", "content": "s"}, {"role": "user", "content": [{"type": "text", "text": "look"}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA", "detail": "low"}}, {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,BBBB"}}]}, {"role": "assistant", "content": "", "tool_calls": [{"id": "t", "type": "function", "function": {"name": "f", "arguments": "{}"}}]}, {"role": "tool", "tool_call_id": "t", "content": [{"type": "text", "text": "shot:"}, {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,CCCC"}}]}]
    )).ok;
    try std.testing.expect(hasVisual(messages));
    var sources: std.ArrayList(@import("vision.zig").Source) = .empty;
    const template = try splitMedia(&cx, messages, .{}, &sources);
    try std.testing.expectEqual(@as(usize, 3), sources.items.len);
    try std.testing.expect(sources.items[0].kind == .image and std.mem.eql(u8, sources.items[0].detail, "low"));
    try std.testing.expect(sources.items[1].kind == .video);
    try std.testing.expect(sources.items[2].kind == .image and std.mem.eql(u8, sources.items[2].detail, "auto"));
    const normalized = try normalizeWith(&cx, template, "system", false, true);
    try std.testing.expectEqualStrings(
        \\[{"role": "system", "content": "s"}, {"role": "user", "content": [{"type": "text", "text": "look"}, {"type": "image", "detail": "low"}, {"type": "video"}]}, {"role": "assistant", "content": "", "tool_calls": [{"id": "t", "type": "function", "function": {"name": "f", "arguments": "{}"}}]}, {"role": "tool", "tool_call_id": "t", "content": [{"type": "text", "text": "shot:"}, {"type": "image", "detail": "auto"}]}]
    , try json.stringify(cx.a, normalized, .{ .ascii = false }));
}

test "split_images refuses what Python refuses" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const cases = [_][2][]const u8{
        .{ "[{\"role\": \"assistant\", \"content\": [{\"type\": \"image_url\", \"image_url\": {\"url\": \"data:image/png;base64,AA\"}}]}]", "image_url parts are supported only in user and tool messages" },
        .{ "[{\"role\": \"tool\", \"content\": [{\"type\": \"video_url\", \"video_url\": {\"url\": \"data:video/mp4;base64,AA\"}}]}]", "video_url parts are supported only in user messages" },
        .{ "[{\"role\": \"user\", \"content\": [{\"type\": \"image_url\", \"image_url\": {\"url\": \"http://x/y.png\"}}]}]", "images require data URLs or public HTTPS URLs" },
        .{ "[{\"role\": \"user\", \"content\": [{\"type\": \"image_url\", \"image_url\": {\"url\": \"https://x/y.png\"}}]}]", "image URLs are off on this server; send the image as a data URL, or start the server with --vision-urls" },
        .{ "[{\"role\": \"user\", \"content\": [{\"type\": \"image_url\", \"image_url\": {\"url\": \"data:,x\", \"detail\": \"max\"}}]}]", "image detail must be auto, low or high" },
        .{ "[{\"role\": \"user\", \"content\": [{\"type\": \"input_audio\", \"input_audio\": {}}]}]", "content parts must be text, image_url or video_url; audio is unsupported" },
    };
    for (cases) |c| {
        var cx: Cx = .{ .a = arena.allocator() };
        var sources: std.ArrayList(@import("vision.zig").Source) = .empty;
        const messages = (try json.parse(cx.a, c[0])).ok;
        try std.testing.expectError(error.Refused, splitMedia(&cx, messages, .{}, &sources));
        try std.testing.expectEqualStrings(c[1], cx.message);
    }
    // more than the request's images
    var cx: Cx = .{ .a = arena.allocator() };
    var sources: std.ArrayList(@import("vision.zig").Source) = .empty;
    const two = (try json.parse(cx.a, "[{\"role\": \"user\", \"content\": [{\"type\": \"image_url\", \"image_url\": {\"url\": \"data:,a\"}}, {\"type\": \"image_url\", \"image_url\": {\"url\": \"data:,b\"}}]}]")).ok;
    try std.testing.expectError(error.Refused, splitMedia(&cx, two, .{ .max_images = 1 }, &sources));
    try std.testing.expect(std.mem.startsWith(u8, cx.message, "a request supports at most 1 images"));
}
