const std = @import("std");
const zg = @import("zigrad");
const MirageKernel = @import("mirage_rt.zig").MirageKernel;

pub fn main() !void {
    const T = f32;
    std.debug.print("Testing Mirage kernel...\n", .{});
    zg.global_graph_init(std.heap.smp_allocator, .{ .eager_teardown = true });
    var cuda = zg.device.CudaDevice.init(0);
    defer cuda.deinit();

    // load kernel
    const M: usize = 16;
    const N: usize = 64;
    const K: usize = 128;

    var kernel = try MirageKernel.load("./matmul_16x64x128.so", 1024 * 1024);
    defer kernel.deinit();

    // check for workspace size?
    const workspace = try cuda.mem_alloc(u8, kernel.workspace_size);
    defer cuda.mem_free(workspace);

    var a_gpu = try zg.NDTensor(T).sequence(cuda.reference(), 0, 1, &.{ M, K }, .{ .label = "A" });
    defer a_gpu.deinit();
    const b_gpu = try zg.NDTensor(T).sequence(cuda.reference(), @floatFromInt(a_gpu.get_size()), 1, &.{ K, N }, .{ .label = "B" });
    defer b_gpu.deinit();

    const c_gpu = try zg.NDTensor(T).zeros(cuda.reference(), &.{ M, N }, .{ .label = "C" });
    defer c_gpu.deinit();

    a_gpu.print();
    std.debug.print("\n\n", .{});
    b_gpu.print();
    std.debug.print("\n\n", .{});
    c_gpu.print();
    std.debug.print("\n\n", .{});

    // Execute
    var inputs = [_]?*const anyopaque{ a_gpu.get_data().ptr, b_gpu.get_data().ptr };
    var outputs = [_]?*anyopaque{c_gpu.get_data().ptr};
    cuda.sync();
    kernel.execute(
        &inputs,
        &outputs,
        workspace.ptr,
        cuda.context.stream.ptr,
    );
    cuda.sync();
    std.debug.print("Executed Mirage kernel.\n", .{});

    std.debug.print("Transferring to host...", .{});
    var cpu = zg.device.HostDevice.init();
    defer cpu.deinit();
    const c_cpu = try c_gpu.to_device(cpu.reference());
    cuda.sync();
    std.debug.print("Done.", .{});
    c_cpu.print();
}

/// Do the same thing but using only the device API
fn test_raw(
    T: type,
    M: usize,
    N: usize,
    K: usize,
    kernel: MirageKernel,
    workspace: *anyopaque,
    cuda: zg.device.CudaDevice,
) !void {
    const a_gpu = try cuda.mem_alloc(T, M * K);
    const b_gpu = try cuda.mem_alloc(T, K * N);
    // not supported for <f32 rn
    // cuda.mem_sequence(T, a_gpu, 0, 1);
    // cuda.mem_sequence(T, b_gpu, M * K, 1);

    const c_gpu = try cuda.mem_alloc(T, M * N);
    // not supported for <f32 rn
    // cuda.mem_fill(T, c_gpu, 0);

    defer {
        cuda.mem_free(a_gpu);
        cuda.mem_free(b_gpu);
        cuda.mem_free(c_gpu);
    }

    var inputs = [_]?*const anyopaque{ a_gpu.ptr, b_gpu.ptr };
    var outputs = [_]?*anyopaque{c_gpu.ptr};
    cuda.sync();
    kernel.execute(
        &inputs,
        &outputs,
        workspace.ptr,
        cuda.context.stream.ptr,
    );
    cuda.sync();
    std.debug.print("Executed Mirage kernel.\n", .{});

    // Not supported for <f32 rn (specifically zeros/mem_fill and print)
    //  still could do it manually without filling with zeros
    std.debug.print("Transferring to host...", .{});
    var cpu = zg.device.HostDevice.init();
    defer cpu.deinit();
    const c_cpu = try cpu.mem_alloc(T, M * N);
    cuda.mem_transfer(T, c_gpu, c_cpu, .DtoH);
    cuda.sync();
    std.debug.print("Done.", .{});
    // const c_cpu_ndarray = try zg.NDArray(T).from_slice(c_cpu, &.{M * N}, cpu.reference());
    // c_cpu_ndarray.print(cpu.reference());
}
