# 换数据集训练

参考：https://blog.csdn.net/qq_43487391/article/details/103927220

======》需要增加【自己的数据集类文件】 并 【加入数据集工厂】：

修改配置文件，在src/lib/datasets/dataset里面新建一个“ped. py”，文件内容先把文件夹下coco.py全部复制过来再进行修改

a：类COCO改为自己设置的类名
b：15行num_classes=80改成自己的类别数
c：17行的mean和std改成自己图片数据集的均值和标准差ped
d：**修改数据和图片路径**，data_dir 输入的是咱们之前建立的数据集文件夹的名字，img_dir 输入的是 images 图片文件夹【可以看到coco数据集是根据train test val保存图片】，super后的类改为自己的类名
e：修改\_\_init\_\_函数中读取Json的文件路径，并且把test的也加上，后续测试时需要用到，根据train test val的不同读取不同的json文件
f：类别名和类别id。这里得根据生成的json文件里面的类别和id一一对应，不然后续测试的结果类别会弄混。

```python
self.class_name = ['__background__', '0', '1',  # 这里就不起名字了，用id代替
                    '2', '3', '4']
self._valid_ids = [0, 1, 2, 3, 4]
```

将数据集加入src/lib/datasets/dataset_factory.py里面，导入自己的类以及在dataset_factory字典里加入自己的数据集名字键值对。

```python
dataset_factory = {
  'coco': COCO,
  'pascal': PascalVOC,
  'kitti': KITTI,
  'coco_hp': COCOHP
   # 加入自己的数据集类
}
```



=======》需要改动：

为了训练自己的数据集，需要修改代码。
修改/src/lib/opts.py

a：将自己的数据集设为默认数据集，加入到help里面

```python
self.parser.add_argument('--dataset', default='ped',
                         help='coco | kitti | coco_hp | pascal | ped')
```

b：修改ctdet任务使用的默认数据集改为新添加的数据集，修改分辨率，类别数，均值，标准差，数据集名字

```python
  def init(self, args=''):
    default_dataset_info = {
      'ctdet': {'default_resolution': [512, 512], 'num_classes': 6,
                'mean': [0.304, 0.300, 0.298], 'std': [0.326, 0.326, 0.325],
                'dataset': 'ped'},
```

c：可以修改--batch_size参数，出现内存溢出的报错，就把batchsize改小一点

6、修改src/lib/utils/debugger.py文件(变成自己数据的类别和名字，前后数据集名字一定保持一致）
a：46行加上这两行

```python
elif num_classes == 6 or dataset == 'ped':
  self.names = ped_class_name
```
b：460行加上自己的类（不要背景）

```python
ped_class_name = [
            'openwindow', 'no_brake', 'opendoor', 'sit_2', 'stand_2',
            'leave_2'] # 我就还是用[‘0’, ‘1’, ‘2’, ‘3’, ‘4’]代替名字了
```

# 其他错误

注意：读取图片时候的NoneType问题

不知道为啥读取图片路径时不对？？？？？..点来点去的太复杂了路径？？opt文件中：

```python
opt.root_dir = os.path.join(os.path.dirname(__file__), '..', '..')
```



临时解决方法：

在datasets/sample/ctdet.py文件修改：

原来：

```python
 def __getitem__(self, index):
    img_id = self.images[index]
    file_name = self.coco.loadImgs(ids=[img_id])[0]['file_name']
    img_path = os.path.join(self.img_dir, file_name)
    ann_ids = self.coco.getAnnIds(imgIds=[img_id])
    anns = self.coco.loadAnns(ids=ann_ids)
    num_objs = min(len(anns), self.max_objs)

    img = cv2.imread(img_path)
```

理由：报错原因是没有加载到图片，图片的路径不对，原来的self.img_dir定义了一个路径，但与本机的不符，所以在上面加一行，重新定义一下图片路径。

img_path之前修改为：

```python
file_name = self.coco.loadImgs(ids=[img_id])[0]['file_name']
self.img_dir = 'F:/CenterNet/data/CheckDataSet/images/'
img_path = os.path.join(self.img_dir, file_name)
```

test阶段，第32行，107行左右，也改为：

```python
img_id = self.images[index]
img_info = self.load_image_func(ids=[img_id])[0]['file_name']
# print(img_info)
self.img_dir = 'F:/CenterNet/data/CheckDataSet/images/'
img_path = os.path.join(self.img_dir, img_info)
```



4？？？、测试的时候，运行test时，会将网络预测的测试集图片的结果保存为results.json ，然后与标签test.json进行比较来计算AP，然后我们发现，results里面目标的中心点坐标就为框的中心点坐标，而之前生成的test里面框的中心点坐标实际为框的左上角坐标，因此重新生成一下test.json，修改其标签信息与results相对应才可。



# demo.py流程

```python
#demo.py
Detector = detector_factory[opt.task] # ctdet
detector = Detector(opt)
ret = detector.run(image_name)

#detectors/ctdet.py
## base_detector.py run()
output, dets, forward_time = self.process(images,return_time=True)
if opt.debug >= 1:
	self.show_result(debugger, image, results)
	self.save_result(image_or_path_or_tensor, result, '/data1/lirui/temp')
## base_detector.py save_result()
## 在这里面保存图像或结果到json

# show_result() # 展示结果

```





# CenterNet（一）论文解读

https://www.jianshu.com/p/0637a1ef46e1

# CenterNet（二）论文解读

https://www.jianshu.com/p/7dc88493a31f