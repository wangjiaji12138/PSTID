"""
PSTID: 原型时空身份网络

继承 STID 的基础架构，在输入嵌入和 MLP 之间加入 PSTID 的原型模块。
通过原型模块将节点嵌入归类成空间码本信息，将时间嵌入归类成时间码本信息，
减少噪声并提升模型的可解释性。
"""

from models.PSTID.PSTID import PSTID

__all__ = ['PSTID']
