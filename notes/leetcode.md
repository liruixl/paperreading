## 206 反转列表

迭代 递归，见技巧总结

## 141 142环形链表 160 相交链表

技巧

## 82 83 删除排序链表重复元素



## *76最小覆盖字串

滑动窗口，抽象伪代码，需要注意的是right的开闭状态

```c++
int left = 0, right = 0;

while (right < s.size()) {
    window.add(s[right]);
    right++; //[left, right)
    
    while (valid) {
        window.remove(s[left]);
        left++;
    }
    
    //right++ //[left, right]
}
```



## 19删除倒数第N个结点

添加哑节点

1、第一次遍历得到链表长度，第二次便利删除第（L-n+1）个结点。

2、双指针，第一个指针向前先移动n+1步，然后一起移动直到第一个指针达到尾部的NULL。

## 111二叉树最小深度

BFS（截至条件？）、递归

## 10正则表达式匹配

动态规划，总是想着有 -1， -1 的坐标，这样不用再考虑越界。

9次提交，考虑有空的情况，给s，p字符串添加头字符“#”，并且初始化行为1，列为1的dp数组。

错误一：= 写成 ==

错误二：未考虑“.\*”的情况，即\*号之前有“.”

错误三：a\*的情况只考虑了匹配多个a，也要考虑一个字符也不匹配的情况。a与aa*的例子。如果只考虑了匹配多个啊，那么问题就变成a与aa的匹配问题`dp[r][c] = dp[r][c-1]`，即a与“”空的匹配结果，结果错误。也就是说这两种情况有一种是匹配的那么就是匹配了。



有时间再看看题解。

官方题解自底向上倒着动规，自顶向下dp递归。

## 14最长公共前缀

想着是简单，写不出来。

## 15三数之和

数组中找到三个数其和为0，想着用dfs来解决，超出时间限制。

记忆中跟39 组合总和差不多，原来39题没有限制是三个数，代码如下：

```c++
vector<vector<int>> combinationSum(vector<int>& candidates, int target) {     
		vector<int> oneSolution;
        vector<vector<int>> result;

        dfs(0,candidates, oneSolution,result,target);

        return result;
    }


    void dfs(int zonghe, const vector<int>& candidates , 
             vector<int>& oneSolution, vector<vector<int>>& result, int target)
    {
        if(zonghe == target)
        {
            result.push_back(oneSolution);
            return;
        }

        for(int num : candidates)
        {
            if(zonghe + num <= target && 
               (oneSolution.empty() || 
                //下面这行代码之所以可以这么写，
                //是因为给出的数组中无重复元素
                //只要保证下一个数字大于等于前一个数字
                //就不会出现重复元组
                //[2,2,3],[3,2,2] = 7
               num>= oneSolution.at(oneSolution.size() - 1)))
            {
                oneSolution.push_back(num);
                dfs(zonghe + num, candidates,oneSolution,result,target);
                oneSolution.pop_back();
            }
        }
    }
}
```

回看1两数之和，使用哈希表存储值与索引的关系。

## [28. 实现 strStr()](https://leetcode-cn.com/problems/implement-strstr/)

返回一字符串在另一个字符串首先出现的位置。

c++中是find函数。

拓展Sunday、KMP算法。不懂

## 35 搜索插入位置

二分查找。以及34题：查找元素的第一个和最后一个位置。见做题技巧-二分查找。

## 30. 串联所有单词的子串=76

滑动窗口，看给出单词个数与滑动窗口中单词个数是否一致。

我用的76的方法。if之后一定要想到else，看是否需要else逻辑。

## 79 单词搜索 回溯

回溯的起点结点怎么找