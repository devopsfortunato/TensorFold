//! libtfgrammar: structured output's optional runtime library (zig/src/grammar/tf_grammar.cc over xgrammar's C++
//! library). `zig build grammar -Dxgrammar=DIR` builds it from an xgrammar checkout at the pinned tag into
//! zig-out/native/lib; the server loads it when the first structured request arrives and serves text without it.

const std = @import("std");

/// The xgrammar release the shim is written and checked against (its masks against the Python package's).
pub const pinned = "v0.2.8";

pub fn steps(b: *std.Build, target: std.Build.ResolvedTarget) void {
    const step = b.step("grammar", "libtfgrammar (structured output) from an xgrammar " ++ pinned ++ " checkout: -Dxgrammar=DIR");
    const test_step = b.step("grammar-test", "libtfgrammar's C ABI through lanes/grammar.zig (zig/src/grammar/tf_grammar_test.zig): -Dxgrammar=DIR");
    const dir = b.option([]const u8, "xgrammar", "An xgrammar " ++ pinned ++ " checkout with 3rdparty/picojson and 3rdparty/dlpack, for `zig build grammar`") orelse {
        const fail = b.addFail("zig build grammar needs -Dxgrammar=DIR: git clone --branch " ++ pinned ++ " https://github.com/mlc-ai/xgrammar, then git submodule update --init 3rdparty/dlpack (picojson is in the tree)");
        step.dependOn(&fail.step);
        test_step.dependOn(&fail.step);
        return;
    };
    const root: std.Build.LazyPath = .{ .cwd_relative = dir };
    const mod = b.createModule(.{ .target = target, .optimize = .ReleaseFast, .link_libcpp = true });
    const flags = [_][]const u8{ "-std=c++17", "-DXGRAMMAR_ENABLE_CPPTRACE=0", "-fexceptions", "-Wno-everything" };
    mod.addIncludePath(root.path(b, "include"));
    mod.addIncludePath(root.path(b, "3rdparty/picojson"));
    mod.addIncludePath(root.path(b, "3rdparty/dlpack/include"));
    mod.addCSourceFiles(.{ .root = root, .files = sources(b, dir), .flags = &flags });
    mod.addCSourceFile(.{ .file = b.path("zig/src/grammar/tf_grammar.cc"), .flags = &flags });
    const lib = b.addLibrary(.{ .name = "tfgrammar", .linkage = .dynamic, .root_module = mod });
    const install = b.addInstallArtifact(lib, .{ .dest_dir = .{ .override = .{ .custom = "native/lib" } } });
    step.dependOn(&install.step);
    // the FFI's tests load the library just installed, as the server does (dlopen of libtfgrammar.so)
    const lanes = b.createModule(.{ .root_source_file = b.path("zig/src/core/lanes/lanes.zig"), .target = target, .optimize = .Debug, .link_libc = true });
    const tests = b.addTest(.{ .root_module = b.createModule(.{ .root_source_file = b.path("zig/src/grammar/tf_grammar_test.zig"), .target = target, .optimize = .Debug, .link_libc = true, .imports = &.{.{ .name = "lanes", .module = lanes }} }) });
    const run = b.addSystemCommand(&.{ "sh", "-c", "LD_LIBRARY_PATH=\"$0\" exec \"$1\"" });
    run.addDirectoryArg(lib.getEmittedBinDirectory());
    run.addArtifactArg(tests);
    test_step.dependOn(&run.step);
}

/// xgrammar's library sources as its CMakeLists globs them: cpp/**/*.cc without the Python bindings (cpp/tvm_ffi).
fn sources(b: *std.Build, dir: []const u8) []const []const u8 {
    var out: std.ArrayList([]const u8) = .empty;
    const io = b.graph.io;
    var cpp = std.Io.Dir.cwd().openDir(io, b.pathJoin(&.{ dir, "cpp" }), .{ .iterate = true }) catch return &.{};
    defer cpp.close(io);
    var walk = cpp.walk(b.allocator) catch @panic("OOM");
    defer walk.deinit();
    while (walk.next(io) catch null) |e| {
        if (e.kind != .file or !std.mem.endsWith(u8, e.path, ".cc") or std.mem.startsWith(u8, e.path, "tvm_ffi")) continue;
        out.append(b.allocator, b.pathJoin(&.{ "cpp", e.path })) catch @panic("OOM");
    }
    std.mem.sort([]const u8, out.items, {}, struct {
        fn less(_: void, x: []const u8, y: []const u8) bool {
            return std.mem.lessThan(u8, x, y);
        }
    }.less);
    return out.items;
}
