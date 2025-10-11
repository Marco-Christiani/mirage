const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const module = b.addModule("main", .{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    const exe = b.addExecutable(.{
        .name = "main",
        .root_module = module,
    });

    // -------------------------------------------
    const enable_cuda = b.option(bool, "enable_cuda", "Enable CUDA") orelse true;
    const rebuild_cuda = b.option(bool, "rebuild_cuda", "Rebuild CUDA") orelse false;
    const enable_mkl = b.option(bool, "enable_mkl", "Enable MKL") orelse false;
    const log_level = b.option(std.log.Level, "log_level", "Log level") orelse .info;

    const zigrad_dep = b.dependency("zigrad", .{
        .target = target,
        .optimize = optimize,
        .enable_mkl = enable_mkl,
        .log_level = log_level,
        .enable_cuda = enable_cuda,
        .rebuild_cuda = rebuild_cuda,
    });
    module.addImport("zigrad", zigrad_dep.module("zigrad"));
    // -------------------------------------------

    exe.addIncludePath(.{ .cwd_relative = "../include/" });
    exe.linkLibCpp();
    b.installArtifact(exe);
}
