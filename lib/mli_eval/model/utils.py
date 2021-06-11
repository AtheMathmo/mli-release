class SumMetricsContainer:
    def __init__(self):
        self.metrics = {}

    def update(self, metrics):
        for key in metrics.keys():
            if key not in self.metrics:
                self.metrics[key] = 0.0
            self.metrics[key] += metrics[key]
  
    def into_avg(self):
        if "n" not in self.metrics:
            raise ValueError("Sum metrics container does not have an 'n' counter field")
        n = self.metrics["n"]
        ret = {}
        for k in self.metrics:
            if k != "n":
                ret[k] = self.metrics[k] / n
        return ret
