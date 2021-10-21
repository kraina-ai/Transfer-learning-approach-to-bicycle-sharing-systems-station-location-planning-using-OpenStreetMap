from sklearn.ensemble import RandomForestClassifier

class BalancedRandomForestClassifier(RandomForestClassifier):
    def __init__(self):
        super().__init__(class_weight="balanced")

class BalancedSubsampleRandomForestClassifier(RandomForestClassifier):
    def __init__(self):
        super().__init__(class_weight="balanced_subsample")