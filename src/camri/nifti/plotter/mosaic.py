import numpy as np
from ..plane import Plane
from ..selector import ImageSelector
from ..handler import Nifti1Handler
from .slice import SlicePlotter

import matplotlib.pyplot as plt
from math import ceil, sqrt
from typing import Optional, Union, List

# Assuming Plane is an enum or class defined elsewhere in the project
from ..plane import Plane  # Update the import path as needed

# 기존에 정의된 Plane, SlicePlotter, ImagePicker 등의 클래스가 있다고 가정합니다.

class MosaicPlotter:
    """
    Class to create a mosaic plot of 3D NIfTI images along a specified plane.
    
    Parameters:
        picker (ImagePicker): An image picker that provides image data and slicing functionality.
        mosaic_plane (str | int | Plane): The plane along which to extract slices 
            (e.g., 'axial', 'coronal', or 'sagittal'). Default is 'axial'.
        slice_indices (List[int], optional): List of slice indices to display.
            If None, slices are evenly sampled.
        clim (tuple, optional): Contrast limits (vmin, vmax) for displaying images.
            If None, computed from the picker data.
        cmap (str): Colormap to use for displaying images.
    """
    def __init__(self,
                 picker,   # type: ImagePicker
                 mosaic_plane: Union[str, int, 'Plane'] = 'axial',
                 slice_indices: Optional[List[int]] = None,
                 clim: Optional[tuple] = None,
                 cmap: str = 'gray'):
        self.picker = picker
        # mosaic_plane를 Plane enum으로 변환 (Plane 클래스에 _parse 혹은 from_string/from_value를 이용)
        self.mosaic_plane = mosaic_plane if hasattr(mosaic_plane, 'value') else Plane.from_string(mosaic_plane) if isinstance(mosaic_plane, str) else Plane.from_value(mosaic_plane)
        self.slice_indices = slice_indices  # 지정하지 않으면 _update_slices에서 자동 샘플링
        self._clim = clim
        self.cmap = cmap
        self._update_slices()
    
    @property
    def clim(self):
        if self._clim is None:
            self._clim = (self.picker.data.min(), self.picker.data.max())
        return self._clim
    
    def _update_slices(self):
        """
        Update the list of slices and their extents for the mosaic.
        슬라이스 인덱스가 지정되지 않았다면, mosaic_plane에 따른 전체 슬라이스 수에서
        최대 16개의 슬라이스를 균등하게 샘플링합니다.
        """
        # mosaic_plane의 축 인덱스 (예: axial이면 2번 축)
        plane_dim = self.mosaic_plane.value  
        shape = self.picker.data.shape  # 3D 또는 4D 데이터라고 가정
        n_slices = shape[plane_dim]
        if self.slice_indices is None:
            # 최대 16개 정도의 슬라이스를 균등하게 선택
            num_to_plot = min(16, n_slices)
            step = max(1, n_slices // num_to_plot)
            self.slice_indices = list(range(0, n_slices, step))
        
        self.slices = []   # 각 슬라이스 이미지 데이터를 저장
        self.extents = []  # 각 슬라이스의 extent (이미지 좌표 범위)
        for idx in self.slice_indices:
            # picker.get_slice는 mosaic_plane과 slice index, coordinate_system='index'를 받아 슬라이스와 extent 반환
            slice_img, extent = self.picker.get_slice(self.mosaic_plane, idx, coordinate_system='index')
            self.slices.append(slice_img)
            self.extents.append(extent)
    
    def plot(self, fig: Optional[plt.Figure] = None, nrows: Optional[int] = None, ncols: Optional[int] = None) -> plt.Figure:
        """
        Create a mosaic plot in a grid of subplots.
        
        Parameters:
            fig (matplotlib.figure.Figure, optional): Figure to use. If None, creates a new one.
            nrows (int, optional): Number of rows in the mosaic grid.
            ncols (int, optional): Number of columns in the mosaic grid.
        
        Returns:
            matplotlib.figure.Figure: The Figure object containing the mosaic.
        """
        num_slices = len(self.slices)
        # 자동으로 grid layout 결정 (가능한 한 정사각형에 가깝게)
        if nrows is None or ncols is None:
            ncols = int(ceil(sqrt(num_slices)))
            nrows = int(ceil(num_slices / ncols))
        
        if fig is None:
            fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3))
        else:
            axes = fig.get_axes()
        
        # axes가 2D array인 경우 flatten
        axes = np.array(axes).flatten()
        
        # 각 슬라이스에 대해 plot 수행
        for ax, slice_img, extent in zip(axes, self.slices, self.extents):
            # SlicePlotter는 각 Axes에 이미지와 extent, contrast limits, cmap 등을 설정해주는 객체라 가정합니다.
            sp = SlicePlotter(ax, slice_img, extent, self.clim, self.cmap)
            sp.plot()
        
        # 남는 subplot은 숨김 처리
        for ax in axes[num_slices:]:
            ax.axis('off')
        
        plt.tight_layout()
        return fig
    
    def update(self, slice_indices: Optional[List[int]] = None, clim: Optional[tuple] = None, cmap: Optional[str] = None):
        """
        Update the mosaic settings and re-plot the mosaic.
        
        Parameters:
            slice_indices (List[int], optional): New slice indices to display.
            clim (tuple, optional): New contrast limits.
            cmap (str, optional): New colormap.
        """
        if slice_indices is not None:
            self.slice_indices = slice_indices
        if clim is not None:
            self._clim = clim
        if cmap is not None:
            self.cmap = cmap
        self._update_slices()
        # 추후 기존 figure에 업데이트할 수 있도록 확장 가능 (여기서는 plot()을 다시 호출하는 방식)
