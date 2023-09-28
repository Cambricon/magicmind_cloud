import abc
import numpy as np
class Metric(metaclass=abc.ABCMeta):
    def __init__(self):
        pass

    @abc.abstractmethod
    def reset(self):
        raise NotImplementedError(
            "function 'reset' not implemented in {}.".format(
                self.__class__.__name__
            )
        )

    @abc.abstractmethod
    def update(self, *args):
        raise NotImplementedError(
            "function 'update' not implemented in {}.".format(
                self.__class__.__name__
            )
        )

    @abc.abstractmethod
    def accumulate(self):
        raise NotImplementedError(
            "function 'accumulate' not implemented in {}.".format(
                self.__class__.__name__
            )
        )

    @abc.abstractmethod
    def name(self):
        raise NotImplementedError(
            "function 'name' not implemented in {}.".format(
                self.__class__.__name__
            )
        )

    def compute(self, *args):
        return args




class Auc(Metric):
    def __init__(
        self, curve='ROC', num_thresholds=4095, name='auc', *args, **kwargs
    ):
        super(Auc, self).__init__(*args, **kwargs)
        self._curve = curve
        self._num_thresholds = num_thresholds

        _num_pred_buckets = num_thresholds + 1
        self._stat_pos = np.zeros(_num_pred_buckets)
        self._stat_neg = np.zeros(_num_pred_buckets)
        self._name = name

    def update(self, preds, labels):
        for i, lbl in enumerate(labels):
            value = preds[i, 1]
            bin_idx = int(value * self._num_thresholds)
            assert bin_idx <= self._num_thresholds
            if lbl:
                self._stat_pos[bin_idx] += 1.0
            else:
                self._stat_neg[bin_idx] += 1.0

    @staticmethod
    def trapezoid_area(x1, x2, y1, y2):
        return abs(x1 - x2) * (y1 + y2) / 2.0

    def accumulate(self):
        tot_pos = 0.0
        tot_neg = 0.0
        auc = 0.0

        idx = self._num_thresholds
        while idx >= 0:
            tot_pos_prev = tot_pos
            tot_neg_prev = tot_neg
            tot_pos += self._stat_pos[idx]
            tot_neg += self._stat_neg[idx]
            auc += self.trapezoid_area(
                tot_neg, tot_neg_prev, tot_pos, tot_pos_prev
            )
            idx -= 1

        return (
            auc / tot_pos / tot_neg if tot_pos > 0.0 and tot_neg > 0.0 else 0.0
        )

    def reset(self):
        _num_pred_buckets = self._num_thresholds + 1
        self._stat_pos = np.zeros(_num_pred_buckets)
        self._stat_neg = np.zeros(_num_pred_buckets)

    def name(self):
        return self._name
