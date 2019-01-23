# 如何使用TensorFlow中的高级API

## Estimator、Experiment和Dataset

Experiment、Estimator 和 DataSet 框架和它们的相互作用 ：

![img](assets/14159image (2).png)

在本文中，我们使用 MNIST 作为数据集。它是一个易于使用的数据集，可以通过 TensorFlow 访问。你可以在这个 gist 中找到完整的示例代码。使用这些框架的一个好处是我们不需要直接处理图形和会话。 

## Estimator

Estimator（评估器）类代表一个模型，以及这些模型被训练和评估的方式。我们可以这样构建一个评估器： 

```python
return tf.estimator.Estimator(
    model_fn=model_fn,  # First-class function
    params=params,  # HParams
    config=run_config  # RunConfig
)
```

为了构建一个 Estimator，我们需要传递一个模型函数，一个参数集合以及一些配置。

+ 参数应该是模型超参数的集合，它可以是一个字典，但我们将在本示例中将其表示为 HParams 对象，用作 namedtuple。
+ 该配置指定如何运行训练和评估，以及如何存出结果。这些配置通过 RunConfig 对象表示，该对象传达 Estimator 需要了解的关于运行模型的环境的所有内容。
+ 模型函数是一个 Python 函数，它构建了给定输入的模型（见后文）。

## 模型函数

模型函数是一个 Python 函数，它作为第一级函数传递给 Estimator。稍后我们就会看到，TensorFlow 也会在其他地方使用第一级函数。模型表示为函数的好处在于模型可以通过实例化函数不断重新构建。该模型可以在训练过程中被不同的输入不断创建，例如：在训练期间运行验证测试。

模型函数将输入特征作为参数，相应标签作为张量。它还有一种模式来标记模型是否正在训练、评估或执行推理。模型函数的最后一个参数是超参数的集合，它们与传递给 Estimator 的内容相同。模型函数需要返回一个 EstimatorSpec 对象——它会定义完整的模型。

EstimatorSpec 接受预测，损失，训练和评估几种操作，因此它定义了用于训练，评估和推理的完整模型图。由于 EstimatorSpec 采用常规 TensorFlow Operations，因此我们可以使用像 TF-Slim 这样的框架来定义自己的模型。



## Experiment

Experiment（实验）类是定义如何训练模型，并将其与 Estimator 进行集成的方式。我们可以这样创建一个实验类： 

```python
experiment = tf.contrib.learn.Experiment(
    estimator=estimator,  # Estimator
    train_input_fn=train_input_fn,  # First-class function
    eval_input_fn=eval_input_fn,  # First-class function
    train_steps=params.train_steps,  # Minibatch steps
    min_eval_frequency=params.min_eval_frequency,  # Eval frequency
    train_monitors=[train_input_hook],  # Hooks for training
    eval_hooks=[eval_input_hook],  # Hooks for evaluation
    eval_steps=None  # Use evaluation feeder until its empty
)
```

Experiment 作为输入：

+ 一个 Estimator（例如上面定义的那个）。
+ 训练和评估数据作为第一级函数。这里用到了和前述模型函数相同的概念，通过传递函数而非操作，如有需要，输入图可以被重建。我们会在后面继续讨论这个概念。
+ 训练和评估钩子（hooks）。这些钩子可以用于监视或保存特定内容，或在图形和会话中进行一些操作。例如，我们将通过操作来帮助初始化数据加载器。
+ 不同参数解释了训练时间和评估时间

一旦我们定义了 experiment，我们就可以通过 learn_runner.run 运行它来训练和评估模型： 

```python
learn_runner.run(
    experiment_fn=experiment_fn,  # First-class function
    run_config=run_config,  # RunConfig
    schedule="train_and_evaluate",  # What to run
    hparams=params  # HParams
)
```

与模型函数和数据函数一样，函数中的学习运算符将创建 experiment 作为参数。 

