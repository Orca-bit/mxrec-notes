# TF优化器

`tensorflow`中优化器类继承自`optimizer.Optimizer` 
```python
def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    ...
```
该方法的默认实现与优化器类型无关，子类可以复写以下方法改变具体行为：
```python
  def _create_slots():
    ...
  def _prepare():
    ...
  def _apply_dense():
    ...
  def _apply_sparse():
    ...
```


## 梯度下降

- 梯度下降方法，计算所有样本梯度的均值
- 随机梯度下降，计算随机采样一个样本的梯度
- 小批量随机梯度下降，采样某个批量大小的样本计算梯度的均值

## 动量法

动量法累积过去的梯度，防止梯度方向剧烈抖动。

### 衰减平均（leaky average）

小批量随机梯度下降平均梯度减小了方差。通过以下方式计算：

\\[\mathbf{g}\_{t, t-1} = \partial\_{\mathbf{w}} \frac{1}{|\mathcal{B}\_t|} \sum_{i \in \mathcal{B}\_t} f(\mathbf{x}\_{i}, \mathbf{w}\_{t-1}) = \frac{1}{|\mathcal{B}\_t|} \sum_{i \in \mathcal{B}\_t} \mathbf{h}\_{i, t-1}.\\]

在这里使用\\(\mathbf{h}\_{i, t-1} = \partial\_{\mathbf{w}} f(\mathbf{x}\_i, \mathbf{w}_{t-1})\\)作为样本\\(i\\)的随机梯度下降，使用时间\\(t-1\\)时更新的权重\\(t-1\\)。
如果能够从方差减少的影响中受益，甚至超过小批量上的梯度平均值，那很不错。
完成这项任务的一种选择是用**衰减平均**（leaky average）取代梯度计算：

\\[\mathbf{v}\_t = \beta \mathbf{v}\_{t-1} + \mathbf{g}\_{t, t-1}\\]

其中\\(\beta \in (0, 1)\\)。
这有效地将瞬时梯度替换为多个“过去”梯度的平均值。
\\(\mathbf{v}\\)被称为**动量**（momentum），
它累加了过去的梯度。
递归地将\\(\mathbf{v}\_t\\)扩展到

\\[\begin{aligned}
\mathbf{v}\_t = \beta^2 \mathbf{v}\_{t-2} + \beta \mathbf{g}\_{t-1, t-2} + \mathbf{g}\_{t, t-1}
= \ldots, = \sum_{\tau = 0}^{t-1} \beta^{\tau} \mathbf{g}\_{t-\tau, t-\tau-1}.
\end{aligned}\\]

其中，较大的\\(\beta\\)相当于长期平均值，而较小的\\(\beta\\)相对于梯度法只是略有修正。
新的梯度替换不再指向特定实例下降最陡的方向，而是指向过去梯度的加权平均值的方向。

### 动量法

**动量法**（momentum）使用\\(\mathbf{v}\_t\\)而不是梯度\\(\mathbf{g}\_t\\)

可以生成以下更新等式：

\\[
\begin{aligned}
\mathbf{v}\_t &\leftarrow \beta \mathbf{v}\_{t-1} + \mathbf{g}\_{t, t-1}, \\\\
\mathbf{x}\_t &\leftarrow \mathbf{x}\_{t-1} - \eta\_t \mathbf{v}\_t.
\end{aligned}
\\]

对于\\(\beta = 0\\)，恢复常规的梯度下降。

## Adagrad

Adaptive Subgradient Methods for Online Learning and Stochastic Optimization:[Duchi et al., 2011](http://jmlr.org/papers/v12/duchi11a.html)([pdf](http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf))

### 算法

解决稀疏场景下不同变量出现频率差异大导致更新不同步的问题。

使用变量\\(\mathbf{s}_t\\)来累加过去的梯度方差，如下所示：

$$\begin{aligned}
    \mathbf{g}\_t & = \partial\_{\mathbf{w}} l(y\_t, f(\mathbf{x}\_t, \mathbf{w})), \\\\
    \mathbf{s}\_t & = \mathbf{s}\_{t-1} + \mathbf{g}\_t^2, \\\\
    \mathbf{w}\_t & = \mathbf{w}\_{t-1} - \frac{\eta}{\sqrt{\mathbf{s}\_t + \epsilon}} \cdot \mathbf{g}\_t.
\end{aligned}$$

根据\\(\mathbf{s}_t\\)的大小来调整学习率，较大梯度的变量会显著缩小，而其他梯度较小的变量则会得到更平滑的处理。

### 实现

adagrad优化器，`mxrec`中的实现与`tensorflow`中的实现基本相同。

[mx_rec/optimizers/adagrad.py · steepcurve/mxrec - Gitee.com](https://gitee.com/steepcurve/mxrec/blob/develop_l00809940/mx_rec/optimizers/adagrad.py)

```python
# MxRec
def _apply_sparse(self, grad, var):
    acc = self.get_slot(var, "acc")
    return training_ops.sparse_apply_adagrad(
        var, acc, math_ops.cast(self._learning_rate_tensor, var.dtype.base_dtype),
        grad.values,
        grad.indices,
        use_locking=self._use_locking)
```

与`tensorflow`不同的地方是创建`slots`时定义的`op`名称
```python
# MxRec
def _create_slots(self, var_list):
    for var in var_list:
        dtype = var.dtype.base_dtype
        if var.get_shape().is_fully_defined():
            init = init_ops.constant_initializer(self._initial_accumulator_value,
                                                 dtype=dtype)
        else:
            init = self._init_constant_op(var, dtype)

        acc_state_name = self._name + "/" + "accumulator"
        self._get_or_make_slot_with_initializer(var, init, var.get_shape(), dtype,
                                                "acc", acc_state_name)

# tensorflow
  def _create_slots(self, var_list):
    for v in var_list:
      dtype = v.dtype.base_dtype
      if v.get_shape().is_fully_defined():
        init = init_ops.constant_initializer(self._initial_accumulator_value,
                                             dtype=dtype)
      else:
        init = self._init_constant_op(v, dtype)
      self._get_or_make_slot_with_initializer(v, init, v.get_shape(), dtype,
                                              "accumulator", self._name)
```

## RMSProp

RMSProp相对于Adagrad所做的改进：

- **学习率衰减问题**：Adagrad通过累积所有梯度的平方到一个状态向量中来调整学习率，这导致随着时间的增长，每个参数的学习率会持续下降，可能过快地变得非常小，从而减慢学习过程，尤其是在优化的后期阶段。RMSProp通过引入一个衰减因子来解决这个问题。
- **衰减平均（Leaky Average）**：RMSProp使用一个衰减平均来代替Adagrad中的累积平均。这意味着每个参数的历史梯度平方会以一个因子进行衰减，从而允许算法“忘记”早期的梯度信息。具体来说，RMSProp的更新规则如下：
$$\begin{aligned}
    \mathbf{s}\_t & \leftarrow \gamma \mathbf{s}\_{t-1} + (1 - \gamma) \mathbf{g}\_t^2, \\\\
    \mathbf{x}\_t & \leftarrow \mathbf{x}\_{t-1} - \frac{\eta}{\sqrt{\mathbf{s}\_t + \epsilon}} \odot \mathbf{g}\_t.
\end{aligned}$$


## Adam

Adam - A Method for Stochastic Optimization: [Kingma et al., 2015](https://arxiv.org/abs/1412.6980) ([pdf](https://arxiv.org/pdf/1412.6980.pdf))

### 算法

Adam算法的关键组成部分之一是：它使用leaky average来估算梯度的动量和二次矩，即它使用状态变量

$$\begin{aligned}
    \mathbf{v}\_t & \leftarrow \beta_1 \mathbf{v}\_{t-1} + (1 - \beta_1) \mathbf{g}\_t, \\\\
    \mathbf{s}\_t & \leftarrow \beta_2 \mathbf{s}\_{t-1} + (1 - \beta_2) \mathbf{g}\_t^2.
\end{aligned}$$

这里\\(\beta_1\\)和\\(\beta_2\\)是非负加权参数。
常将它们设置为\\(\beta_1 = 0.9\\)和\\(\beta_2 = 0.999\\)。
也就是说，二次矩估计的移动远远慢于动量估计的移动。
如果初始化\\(\mathbf{v}\_0 = \mathbf{s}\_0 = 0\\)，就存在相当大的初始偏差。
通过使用\\(\sum_{i=0}^t \beta^i = \frac{1 - \beta^t}{1 - \beta}\\)来解决这个问题。
相应地，标准化状态变量由下式获得

$$\hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1 - \beta_1^t} \text{ and } \hat{\mathbf{s}}_t = \frac{\mathbf{s}_t}{1 - \beta_2^t}.$$

首先，以类似于RMSProp算法的方式重新缩放梯度以获得

$$\mathbf{g}_t' = \frac{\eta \hat{\mathbf{v}}_t}{\sqrt{\hat{\mathbf{s}}_t} + \epsilon}.$$

与RMSProp不同，更新使用动量\\(\hat{\mathbf{v}}_t\\)而不是梯度本身。
此外，使用\\(\frac{1}{\sqrt{\hat{\mathbf{s}}_t} + \epsilon}\\)而不是\\(\frac{1}{\sqrt{\hat{\mathbf{s}}_t + \epsilon}}\\)进行缩放。
通常，选择\\(\epsilon = 10^{-6}\\)，这是为了在数值稳定性和逼真度之间取得良好的平衡。

$$\mathbf{x}\_t \leftarrow \mathbf{x}\_{t-1} - \mathbf{g}\_t'.$$

### 实现

实现中变量表示略有不同，同时约简了部分计算。

初始化

$$m_0 := 0 \text{(Initialize initial 1st moment vector)}$$
$$v_0 := 0 \text{(Initialize initial 2nd moment vector)}$$
$$t := 0 \text{(Initialize timestep)}$$

参数更新        

$$t := t + 1$$

$$\text{lr}_t := \mathrm{learning\\_rate} * \sqrt{1 - \beta_2^t} / (1 - \beta_1^t)$$

$$m_t := \beta_1 * m_{t-1} + (1 - \beta_1) * g$$

$$v_t := \beta_2 * v_{t-1} + (1 - \beta_2) * g^2$$

$$\text{variable} := \text{variable} - \text{lr}_t * m_t / (\sqrt{v_t} + \epsilon)$$
 

`tensorflow`中实现了`Adam`优化器，具体实现逻辑如下：

[tensorflow 源码](https://github.com/tensorflow/tensorflow/blob/80b1605dbc7ac2f475dff03b13d7efcf295d35c4/tensorflow/python/training/adam.py#L247)
```python
def _apply_sparse_shared(self, grad, var, indices, scatter_add):
    # 主要计算逻辑
    beta1_power, beta2_power = self._get_beta_accumulators()
    beta1_power = math_ops.cast(beta1_power, var.dtype.base_dtype)
    beta2_power = math_ops.cast(beta2_power, var.dtype.base_dtype)
    lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
    beta1_t = math_ops.cast(self._beta1_t, var.dtype.base_dtype)
    beta2_t = math_ops.cast(self._beta2_t, var.dtype.base_dtype)
    epsilon_t = math_ops.cast(self._epsilon_t, var.dtype.base_dtype)
    lr = (lr_t * math_ops.sqrt(1 - beta2_power) / (1 - beta1_power))
    # m_t = beta1 * m + (1 - beta1) * g_t
    m = self.get_slot(var, "m")
    m_scaled_g_values = grad * (1 - beta1_t)
    m_t = state_ops.assign(m, m * beta1_t, use_locking=self._use_locking)
    with ops.control_dependencies([m_t]):
        m_t = scatter_add(m, indices, m_scaled_g_values)
    # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
    v = self.get_slot(var, "v")
    v_scaled_g_values = (grad * grad) * (1 - beta2_t)
    v_t = state_ops.assign(v, v * beta2_t, use_locking=self._use_locking)
    with ops.control_dependencies([v_t]):
        v_t = scatter_add(v, indices, v_scaled_g_values)
    v_sqrt = math_ops.sqrt(v_t)
    var_update = state_ops.assign_sub(
        var, lr * m_t / (v_sqrt + epsilon_t), use_locking=self._use_locking)
    return control_flow_ops.group(*[var_update, m_t, v_t])
```

### MxRec中的lazy_adam

[mx_rec/optimizers/lazy_adam.py · Ascend/mxrec - Gitee.com](https://gitee.com/ascend/mxrec/blob/master/mx_rec/optimizers/lazy_adam.py)

```python
    def _apply_sparse_shared(self, grad, var, indices, scatter_nd_add):
	    # 大部分计算流程相同，代码略有差异
        power_b1, power_b2 = self._get_beta_accumulators()
        power_b1 = math_ops.cast(power_b1, var.dtype.base_dtype)
        power_b2 = math_ops.cast(power_b2, var.dtype.base_dtype)
        temp = self._cast_to_base_type(var)
        temp_lr = temp.get("temp_lr")
        temp_b1 = temp.get("temp_b1")
        temp_b2 = temp.get("temp_b2")
        temp_epsilon = temp.get("temp_epsilon")
        learning_rate = tf.divide(temp_lr * math_ops.sqrt(1 - power_b2), (1 - power_b1))
        abs_indices = tf.math.maximum(indices, 0)
        nd_indices = tf.expand_dims(indices, 1)
        momentum = self.get_slot(var, "m")
        old_m_slice = tf.gather(momentum, abs_indices)
        m_t_slice = temp_b1 * old_m_slice + (1 - temp_b1) * grad
        ## DIFF 这里算子与tensorflow中不同
        ## tensorflow中没有gather
        ## 这里相当于计算两次，gather后先计算新的m，再将其与旧的m的差值更新至slot
        m_update_op = scatter_nd_add(momentum, nd_indices, m_t_slice - old_m_slice)
        velocity = self.get_slot(var, "v")
        old_v_slice = tf.gather(velocity, abs_indices)
        v_t_slice = temp_b2 * old_v_slice + (1 - temp_b2) * math_ops.square(grad)
        ## DIFF 同上
        v_update_op = scatter_nd_add(velocity, nd_indices, v_t_slice - old_v_slice)
        denominator_slice = math_ops.sqrt(v_t_slice) + temp_epsilon
        var_update_op = scatter_nd_add(var, nd_indices, tf.divide(-learning_rate * m_t_slice, denominator_slice))
        return control_flow_ops.group(m_update_op, v_update_op, var_update_op)
```

### MxRec中的lazy_adam_by_address

[mx_rec/optimizers/lazy_adam_by_addr.py · Ascend/mxrec - 码云 - 开源中国 (gitee.com)](https://gitee.com/ascend/mxrec/blob/master/mx_rec/optimizers/lazy_adam_by_addr.py)

```python
def _apply_sparse_shared(self, grad, addr):
    power_b1, power_b2 = self._get_beta_accumulators()
    power_b1 = math_ops.cast(power_b1, grad.dtype.base_dtype)
    power_b2 = math_ops.cast(power_b2, grad.dtype.base_dtype)
    temp = self._cast_to_base_type(grad)
    temp_lr = temp.get("temp_lr")
    temp_b1 = temp.get("temp_b1")
    temp_b2 = temp.get("temp_b2")
    temp_epsilon = temp.get("temp_epsilon")
    learning_rate = tf.divide(temp_lr * math_ops.sqrt(1 - power_b2), (1 - power_b1))
    host_pipeline_ops = import_host_pipeline_ops()
    dim = grad.shape.as_list()[-1]
    ## 动态扩容场景，m v没有放在tf的slots
    ## 梯度更新时调用c++侧的算子
    combined_tensor = \
        host_pipeline_ops.embedding_lookup_by_address(addr, embedding_dim=3 * dim, embedding_type=1)
    ## 查询结果中包含模型参数 m v
    split_length = [dim] + [dim] + [dim]
    split_tensors = tf.split(combined_tensor, split_length, axis=1)
    old_m_slice = split_tensors[1]
    m_t_slice = temp_b1 * old_m_slice + (1 - temp_b1) * grad
    old_v_slice = split_tensors[2]
    v_t_slice = temp_b2 * old_v_slice + (1 - temp_b2) * math_ops.square(grad)
    denominator_slice = math_ops.sqrt(v_t_slice) + temp_epsilon
    update_list = [tf.divide(-learning_rate * m_t_slice, denominator_slice)] + [m_t_slice - old_m_slice] + \
                  [v_t_slice - old_v_slice]
    update_tensor = tf.concat(update_list, axis=1)
    ## 调用算子进行模型参数更新
    var_update_op = host_pipeline_ops.embedding_update_by_address(addr, update_tensor, update_type=0)
    return var_update_op
```





