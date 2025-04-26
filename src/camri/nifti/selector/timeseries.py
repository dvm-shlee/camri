class TimeSeriesSelector:
    def __init__(self, viewer):
        self.viewer = viewer
        self._current_timepoint = 0
        self._current_timepoint_index = 0

    def set_timepoint(self, timepoint):
        self._current_timepoint = timepoint
        self._current_timepoint_index = timepoint
        self.viewer.update_view()