# 二分法相关笔记
#### 为什么在二分查找中要避免整数溢出？
- 在二分查找算法中，我们通常会计算中间位置的索引。如果数组长度非常大，比如数组右边界索引 right 是非常大的数（如 10^9），那么直接用 (left + right) // 2 可能会因为 left + right 的和超过了整数类型能表示的范围，导致溢出。
- 因此可以用 mid = left + (right - left) // 2 来计算中间位置，以减少溢出可能性，因为 (right - left) 的值比 left + right 小得多。

#### 为什么在二分查找中算法中，使用 while (left <= right) 而不是 while (left < right)？
- 这是是为了确保在搜索范围只剩一个元素时仍然能够正确检查该元素。
  - 如果使用 while (left < right)，当 left == right 时，循环会终止，导致最后一个元素不会被检查。例如：

例：假设数组是 [1, 3, 5, 7, 9]，目标值是 9。<br>
初始范围：left = 0，right = 4。<br>
计算 mid = 2，nums[2] = 5。<br>
因为 5 < 9，调整 left = 3。<br>
新范围：left = 3，right = 4。<br>
计算 mid = 3，nums[3] = 7。<br>
因为 7 < 9，调整 left = 4。<br>
此时 left = 4，right = 4，left < right 为假，循环终止。<br>
返回 -1，但实际上目标值 9 存在于数组中，导致错误结果。

#### Math.floor
`Math.floor` is a built-in mathematical method in JavaScript. It returns the largest integer less than or equal to a given number. For example, Math.floor(3.7) returns 3, and Math.floor(-1.2) returns -2.
`Math.floor` 是 JavaScript 中的一个内置数学方法。它的作用是返回小于或等于一个给定数字的最大整数。比如，Math.floor(3.7) 的结果是 3，Math.floor(-1.2) 的结果是 -2。

---
