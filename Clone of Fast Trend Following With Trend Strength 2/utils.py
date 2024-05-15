#region imports
from AlgorithmImports import *
#endregion
def GetPositionSize(group):
    subcategories = {}
    for category, subcategory in group.values():
        if category not in subcategories:
            subcategories[category] = {subcategory: 0}
        elif subcategory not in subcategories[category]:
            subcategories[category][subcategory] = 0
        subcategories[category][subcategory] += 1

    category_count = len(subcategories.keys())
    subcategory_count = {category: len(subcategory.keys()) for category, subcategory in subcategories.items()}
    
    weights = {}
    for symbol in group:
        category, subcategory = group[symbol]
        weight = 1 / category_count / subcategory_count[category] / subcategories[category][subcategory]
        weights[symbol] = weight
    
    return weights
