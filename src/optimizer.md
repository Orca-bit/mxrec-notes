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

## ADAM优化器

Adam - A Method for Stochastic Optimization: [Kingma et al., 2015](https://arxiv.org/abs/1412.6980) ([pdf](https://arxiv.org/pdf/1412.6980.pdf))

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

## MxRec中的lazy_adam

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
        ## DIFF 这里update的数据与adam不同？？ m_t_slice - old_m_slice
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