
import numpy as np
class SportModel:
    def __init__(self,pipe,threshold):
        self.pipe=pipe; self.threshold_=threshold; self.lr_pipeline_=pipe
        self.classes_=np.array([0,1])
    def predict_proba(self,X): return self.pipe.predict_proba(X)
    def predict(self,X): return (self.predict_proba(X)[:,1]>=self.threshold_).astype(int)
    @property
    def named_steps(self): return self.pipe.named_steps
