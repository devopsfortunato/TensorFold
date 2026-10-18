//! tf-flashnext-csf SHARD.safetensors PROJ_SCALE_NAME N K: decodes the CSF pair `<NAME>.nvfp4_csf_fixed` and
//! `<NAME>.nvfp4_csf_exceptions` of one expert projection and prints the sha256 of the plain [N, K/16] scale bytes
//! (compare with a Python decode of the same tensors).
const std = @import("std");
const core = @import("core");
const flashnext = @import("flashnext");

pub fn main(init: std.process.Init) !u8 {
    const gpa = init.gpa;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 5) {
        std.debug.print("usage: tf-flashnext-csf SHARD NAME N K\n", .{});
        return 2;
    }
    const n = try std.fmt.parseInt(usize, args[3], 10);
    const k = try std.fmt.parseInt(usize, args[4], 10);
    var f = try core.safetensors.File.open(gpa, init.io, args[1]);
    defer f.close(init.io);
    var nb: [2][400]u8 = undefined;
    const fixed = f.get(try std.fmt.bufPrint(&nb[0], "{s}{s}", .{ args[2], flashnext.csf.fixed_suffix })) orelse return error.MissingTensor;
    const exc = f.get(try std.fmt.bufPrint(&nb[1], "{s}{s}", .{ args[2], flashnext.csf.exceptions_suffix })) orelse return error.MissingTensor;
    const out = try gpa.alloc(u8, flashnext.csf.plainLen(n, k));
    defer gpa.free(out);
    try flashnext.csf.decode(fixed.bytes, exc.bytes, n, k, out);
    var h: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(out, &h, .{});
    std.debug.print("{x} exceptions={d}\n", .{ &h, exc.bytes.len / 4 });
    return 0;
}
