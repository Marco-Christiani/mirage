const std = @import("std");

const c = @cImport({
    @cInclude("dlfcn.h");
});

const ExecuteMugraphFn = *const fn (
    input_ptrs: [*]?*const anyopaque,
    num_inputs: usize,
    output_ptrs: [*]?*anyopaque,
    num_outputs: usize,
    workspace: ?*anyopaque,
    stream: ?*anyopaque,
    profiler: ?*anyopaque,
) callconv(.C) void;

pub const MirageKernel = struct {
    lib_handle: *anyopaque,
    execute_fn: ExecuteMugraphFn,
    workspace_size: usize,

    pub fn load(path: [:0]const u8, workspace_size: usize) !MirageKernel {
        const handle = c.dlopen(path.ptr, c.RTLD_LAZY) orelse {
            std.debug.print("dlopen failed: {s}\n", .{c.dlerror()});
            return error.LoadFailed;
        };

        const func_name = "execute_mugraph_wrapper";
        const func = c.dlsym(handle, func_name) orelse {
            std.debug.print("dlsym failed for {s}: {s}\n", .{ func_name, c.dlerror() });
            _ = c.dlclose(handle);
            return error.SymbolNotFound;
        };

        return .{
            .lib_handle = handle,
            .execute_fn = @ptrCast(@alignCast(func)),
            .workspace_size = workspace_size,
        };
    }

    pub fn deinit(self: *MirageKernel) void {
        _ = c.dlclose(self.lib_handle);
        self.* = undefined;
    }

    pub fn execute(
        self: *const MirageKernel,
        inputs: []?*const anyopaque,
        outputs: []?*anyopaque,
        workspace: ?*anyopaque,
        stream: ?*anyopaque,
    ) void {
        self.execute_fn(
            inputs.ptr,
            inputs.len,
            outputs.ptr,
            outputs.len,
            workspace,
            stream,
            null, // profiler
        );
    }
};

