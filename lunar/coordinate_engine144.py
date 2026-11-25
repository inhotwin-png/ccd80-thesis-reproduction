"""Frozen scalar predictor used by the FeO rank model."""

class ScalarView:
    def __init__(self,model):self.model=model
    def predict(self,b):return self.model.predict(b['raw'] if isinstance(b,dict) else b)
