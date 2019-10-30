public继承意味着**is-a**的关系，所以对base class为真的任何事情一定也对其derived classes为真。

+ pure virtual函数

  在抽象class中没有定义；它们必须被继承了它们的具象class重新申明。

  “你必须提供一个函数，但我不干涉你怎么实现他”。

+ impure virtual函数

  让子类继承接口和缺省实现，但子类可以提供自己的行为。

+ non virtual函数

  由于 non-virtual函数代表的意义是不变性Cinvariant) 凌驾特异性 (specialization) , 所以它绝不该在 derived class 中被重新定义。    

35 考虑virtual以外的其他选择

