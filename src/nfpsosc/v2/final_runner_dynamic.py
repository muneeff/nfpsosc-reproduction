"""
ملحق لـ final_runner.py - يضيف دعم الطريقة الديناميكية
"""

from __future__ import annotations

from .final_runner import *
from .dynamic_runner import DynamicPCNFPSO

# إضافة الطريقة الديناميكية
DYNAMIC_METHOD = "pc_nfpso_dynamic"

def is_dynamic_method(method: str) -> bool:
    """التحقق مما إذا كانت الطريقة ديناميكية"""
    return method == DYNAMIC_METHOD

def create_dynamic_model(
    n_lags: int,
    validation_size: int,
    **kwargs
) -> DynamicPCNFPSO:
    """إنشاء نموذج DynamicPCNFPSO"""
    return DynamicPCNFPSO(
        n_lags=n_lags,
        validation_size=validation_size,
        **kwargs
    )

# يمكن إضافة المزيد من الدوال حسب الحاجة