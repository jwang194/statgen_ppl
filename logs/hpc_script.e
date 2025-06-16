2025-06-14 11:29:57.413733: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
2025-06-14 11:30:02.089869: W external/xla/xla/service/gpu/nvptx_compiler.cc:765] The NVIDIA driver's CUDA version is 12.2 which is older than the ptxas CUDA version (12.9.86). Because the driver is older than the ptxas version, XLA is disabling parallel compilation, which may slow down compilation. You should update your NVIDIA driver or use the NVIDIA-provided CUDA forward compatibility packages.
E0614 12:53:15.113059   81332 pjrt_stream_executor_client.cc:2985] Execution of replica 0 failed: INTERNAL: Failed to complete all kernels launched on stream 0x5b221c0: Could not synchronize CUDA stream: CUDA_ERROR_ILLEGAL_ADDRESS: an illegal memory access was encountered
E0614 12:53:15.113066   81334 pjrt_stream_executor_client.cc:2985] Execution of replica 1 failed: INTERNAL: Failed to complete all kernels launched on stream 0x5c48980: Could not synchronize CUDA stream: CUDA_ERROR_ILLEGAL_ADDRESS: an illegal memory access was encountered
jax.errors.SimplifiedTraceback: For simplicity, JAX has removed its internal frames from the traceback of the following exception. Set JAX_TRACEBACK_FILTERING=off to include these.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/u/project/zaitlenlab/jwang194/zaitlen/statgen_ppl/distribute_benchmark.py", line 55, in <module>
    states, trace = run(random.key(0),sharded_data,pass_data)
jaxlib.xla_extension.XlaRuntimeError: INTERNAL: Failed to complete all kernels launched on stream 0x5b221c0: Could not synchronize CUDA stream: CUDA_ERROR_ILLEGAL_ADDRESS: an illegal memory access was encountered: while running replica 0 and partition 0 of a replicated computation (other replicas may have failed as well).
2025-06-14 12:53:15.963283: E external/xla/xla/stream_executor/cuda/cuda_driver.cc:1595] failed to free device memory at 0x2ba628a00000; result: CUDA_ERROR_ILLEGAL_ADDRESS: an illegal memory access was encountered
2025-06-14 12:53:15.963676: E external/xla/xla/stream_executor/cuda/cuda_driver.cc:1595] failed to free device memory at 0x2ba628a00200; result: CUDA_ERROR_ILLEGAL_ADDRESS: an illegal memory access was encountered
2025-06-14 12:53:15.963684: E external/xla/xla/stream_executor/cuda/cuda_driver.cc:1595] failed to free device memory at 0x2ba628a01200; result: CUDA_ERROR_ILLEGAL_ADDRESS: an illegal memory access was encountered
2025-06-14 12:53:15.963690: E external/xla/xla/stream_executor/cuda/cuda_driver.cc:1595] failed to free device memory at 0x2ba628a03200; result: CUDA_ERROR_ILLEGAL_ADDRESS: an illegal memory access was encountered
2025-06-14 12:53:15.963697: E external/xla/xla/stream_executor/cuda/cuda_driver.cc:1595] failed to free device memory at 0x2ba1bf203400; result: CUDA_ERROR_ILLEGAL_ADDRESS: an illegal memory access was encountered
2025-06-14 12:53:15.963703: E external/xla/xla/stream_executor/cuda/cuda_driver.cc:1595] failed to free device memory at 0x2ba1bf203600; result: CUDA_ERROR_ILLEGAL_ADDRESS: an illegal memory access was encountered
2025-06-14 12:53:15.963709: E external/xla/xla/stream_executor/cuda/cuda_driver.cc:1595] failed to free device memory at 0x2ba1bf204600; result: CUDA_ERROR_ILLEGAL_ADDRESS: an illegal memory access was encountered
2025-06-14 12:53:15.963714: E external/xla/xla/stream_executor/cuda/cuda_driver.cc:1595] failed to free device memory at 0x2ba1bf206600; result: CUDA_ERROR_ILLEGAL_ADDRESS: an illegal memory access was encountered
2025-06-14 12:53:15.976973: E external/xla/xla/stream_executor/cuda/cuda_driver.cc:1652] error deallocating host memory at 0x2ba628d20a00: CUDA_ERROR_ILLEGAL_ADDRESS: an illegal memory access was encountered
2025-06-14 12:53:15.976999: E external/xla/xla/stream_executor/cuda/cuda_driver.cc:1652] error deallocating host memory at 0x2ba1bf520c00: CUDA_ERROR_ILLEGAL_ADDRESS: an illegal memory access was encountered
2025-06-14 12:53:15.977051: E external/xla/xla/stream_executor/cuda/cuda_driver.cc:1652] error deallocating host memory at 0x2ba1bf520a00: CUDA_ERROR_ILLEGAL_ADDRESS: an illegal memory access was encountered
2025-06-14 12:53:15.977063: E external/xla/xla/stream_executor/cuda/cuda_driver.cc:1652] error deallocating host memory at 0x2ba628d20800: CUDA_ERROR_ILLEGAL_ADDRESS: an illegal memory access was encountered
2025-06-14 12:53:15.977173: E external/xla/xla/stream_executor/cuda/cuda_driver.cc:1652] error deallocating host memory at 0x2ba1bf520800: CUDA_ERROR_ILLEGAL_ADDRESS: an illegal memory access was encountered
2025-06-14 12:53:15.977179: E external/xla/xla/stream_executor/cuda/cuda_driver.cc:1652] error deallocating host memory at 0x2ba628d20600: CUDA_ERROR_ILLEGAL_ADDRESS: an illegal memory access was encountered
2025-06-14 12:53:15.977460: E external/xla/xla/stream_executor/cuda/cuda_driver.cc:1652] error deallocating host memory at 0x2ba1bf520600: CUDA_ERROR_ILLEGAL_ADDRESS: an illegal memory access was encountered
2025-06-14 12:53:15.977469: E external/xla/xla/stream_executor/cuda/cuda_driver.cc:1652] error deallocating host memory at 0x2ba628d20400: CUDA_ERROR_ILLEGAL_ADDRESS: an illegal memory access was encountered
2025-06-14 12:53:15.977771: E external/xla/xla/stream_executor/cuda/cuda_driver.cc:1652] error deallocating host memory at 0x2ba1bf520400: CUDA_ERROR_ILLEGAL_ADDRESS: an illegal memory access was encountered
2025-06-14 12:53:15.977780: E external/xla/xla/stream_executor/cuda/cuda_driver.cc:1652] error deallocating host memory at 0x2ba628d20200: CUDA_ERROR_ILLEGAL_ADDRESS: an illegal memory access was encountered
2025-06-14 12:53:15.978051: E external/xla/xla/stream_executor/cuda/cuda_driver.cc:1507] failed to unload module 0x2ba1c03c8ff0; leaking: CUDA_ERROR_ILLEGAL_ADDRESS: an illegal memory access was encountered
2025-06-14 12:53:15.978059: E external/xla/xla/stream_executor/cuda/cuda_driver.cc:1507] failed to unload module 0x2ba1a4b31040; leaking: CUDA_ERROR_ILLEGAL_ADDRESS: an illegal memory access was encountered
Traceback (most recent call last):
  File "/u/project/zaitlenlab/jwang194/zaitlen/statgen_ppl/extractor.py", line 23, in <module>
    runtimes.append((N,M,dt['runtime_%i_GPU%s'%(N_GPU,smart_init_tag)][()]))
  File "h5py/_objects.pyx", line 54, in h5py._objects.with_phil.wrapper
  File "h5py/_objects.pyx", line 55, in h5py._objects.with_phil.wrapper
  File "/u/home/j/jwang194/.local/lib/python3.9/site-packages/h5py/_hl/group.py", line 357, in __getitem__
    oid = h5o.open(self.id, self._e(name), lapl=self._lapl)
  File "h5py/_objects.pyx", line 54, in h5py._objects.with_phil.wrapper
  File "h5py/_objects.pyx", line 55, in h5py._objects.with_phil.wrapper
  File "h5py/h5o.pyx", line 189, in h5py.h5o.open
KeyError: "Unable to synchronously open object (object 'runtime_2_GPU_smart' doesn't exist)"
