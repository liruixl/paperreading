# 问题：

```
InvalidArgumentError (see above for traceback): flat indices[7, :] = [7, -1] does not index into param (shape: [8,8732]).
	 [[Node: post_forward/GatherNd = GatherNd[Tindices=DT_INT32, Tparams=DT_FLOAT, _device="/job:localhost/replica:0/task:0/device:CPU:0"](post_forward/TopKV2, post_forward/stack)]]
```

背景标签必须是0。。。。。。。。。。。。



