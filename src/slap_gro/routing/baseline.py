"""
Baseline Routing Engine for SLAP-GRO.

Implements standard benchmark routing heuristics and evaluates the original baseline
route reported in literature and industrial warehouse operations.

Strictly distinguishes between:
1. 'actual_company_route': The raw sequence of picks recorded by company pickers in Picking_Wave.csv.
2. 'reconstructed_baseline': A standardized baseline reconstructed via deterministic heuristics
   (e.g., FIFO Canonical, S-Shape / Serpentine, Nearest Neighbor).
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Set, Tuple

from slap_gro.picking.picking_points import (
    build_gro_route_targets,
    get_wave_picking_locations,
)
from slap_gro.picking.wave import PickingWave
from slap_gro.routing.distance import distance
from slap_gro.routing.route import Route, evaluate_route
from slap_gro.storage.base import StoragePolicy
from slap_gro.warehouse.location import Location
from slap_gro.warehouse.warehouse_layout import WarehouseLayout

BaselineType = Literal["actual_company_route", "reconstructed_baseline"]


@dataclass
class BaselineResult:
    """
    Kết quả đánh giá một lộ trình cơ sở (Baseline Route) cho một picking wave.

    Attributes:
        wave_id: Mã đợt nhặt hàng.
        baseline_name: Tên của giải thuật cơ sở ('original', 'nearest_neighbor', 's_shape', 'fifo').
        baseline_type: Phân loại cơ sở ('actual_company_route' hoặc 'reconstructed_baseline').
        total_distance: Tổng quãng đường di chuyển (mét).
        stops: Danh sách mã vị trí ghé thăm [DEPOT, L1, L2, ..., DEPOT].
        stops_count: Số lượt lấy hàng (không tính lượt quay về Depot).
        execution_time_seconds: Thời gian tính toán lộ trình (giây).
        description: Mô tả chi tiết về phương pháp cơ sở.
    """

    wave_id: int
    baseline_name: str
    baseline_type: BaselineType
    total_distance: float
    stops: List[str] = field(default_factory=list)
    stops_count: int = 0
    execution_time_seconds: float = 0.0
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Xuất kết quả sang dạng dictionary."""
        d = asdict(self)
        d["stops_str"] = " -> ".join(self.stops) if self.stops else ""
        return d


class BaseRoutingHeuristic(ABC):
    """Lớp cơ sở trừu tượng cho tất cả các giải thuật định tuyến cơ sở (Baselines)."""

    def __init__(
        self,
        name: str,
        baseline_type: BaselineType,
        description: str = "",
    ) -> None:
        self.name = name.lower().strip()
        self.baseline_type = baseline_type
        self.description = description

    @abstractmethod
    def route(
        self,
        depot: Location,
        pick_locations: List[Location],
        distance_fn: Optional[Callable[[Location, Location], float]] = None,
    ) -> List[Location]:
        """
        Xây dựng chuỗi các điểm dừng bắt đầu từ Depot, ghé thăm tất cả pick_locations
        và kết thúc tại Depot: [depot, L1, L2, ..., Ln, depot].
        """
        pass


class OriginalCompanyRouteBaseline(BaseRoutingHeuristic):
    """
    Lộ trình nguyên bản thực tế của công ty (Actual Company Route / Natural Wave Order).

    Giữ nguyên chính xác thứ tự ghé thăm các vị trí lưu trữ như đã được nhân viên
    quét mã và ghi nhận trong nhật ký WMS (Picking_Wave.csv).
    """

    def __init__(self, is_real_data: bool = True) -> None:
        b_type: BaselineType = "actual_company_route" if is_real_data else "reconstructed_baseline"
        desc = (
            "Lộ trình nguyên bản theo nhật ký quét thực tế của công ty (Picking_Wave.csv)"
            if is_real_data
            else "Lộ trình tái dựng theo thứ tự danh sách nhặt hàng ban đầu (Canonical FIFO)"
        )
        super().__init__(name="original", baseline_type=b_type, description=desc)

    def route(
        self,
        depot: Location,
        pick_locations: List[Location],
        distance_fn: Optional[Callable[[Location, Location], float]] = None,
    ) -> List[Location]:
        if not pick_locations:
            return [depot, depot]
        return [depot] + list(pick_locations) + [depot]


class ReconstructedFIFOBaseline(BaseRoutingHeuristic):
    """
    Lộ trình tái dựng tuần tự FIFO (Reconstructed Canonical Baseline).

    Ghé thăm các vị trí theo đúng thứ tự xuất hiện trong danh sách yêu cầu lấy hàng.
    Được đánh dấu tường minh là 'reconstructed_baseline'.
    """

    def __init__(self) -> None:
        super().__init__(
            name="fifo",
            baseline_type="reconstructed_baseline",
            description="Lộ trình tái dựng tuần tự theo thứ tự xuất hiện trong danh sách nhặt (FIFO)",
        )

    def route(
        self,
        depot: Location,
        pick_locations: List[Location],
        distance_fn: Optional[Callable[[Location, Location], float]] = None,
    ) -> List[Location]:
        if not pick_locations:
            return [depot, depot]
        return [depot] + list(pick_locations) + [depot]


class NearestNeighborBaseline(BaseRoutingHeuristic):
    """
    Heuristic láng giềng gần nhất (Greedy Nearest Neighbor Heuristic).

    Bắt đầu tại Depot, tại mỗi bước thuật toán tham lam chọn vị trí lấy hàng gần nhất
    chưa được ghé thăm cho đến khi hoàn thành tất cả các điểm, sau đó quay trở về Depot.
    """

    def __init__(self) -> None:
        super().__init__(
            name="nearest_neighbor",
            baseline_type="reconstructed_baseline",
            description="Heuristic tái dựng: Láng giềng gần nhất (Greedy Nearest Neighbor)",
        )

    def route(
        self,
        depot: Location,
        pick_locations: List[Location],
        distance_fn: Optional[Callable[[Location, Location], float]] = None,
    ) -> List[Location]:
        if not pick_locations:
            return [depot, depot]

        dist_fn = distance_fn or (lambda a, b: a.distance_to(b))

        unvisited = list(pick_locations)
        current = depot
        tour = [depot]

        while unvisited:
            best_idx = 0
            best_dist = float("inf")
            for idx, candidate in enumerate(unvisited):
                d = dist_fn(current, candidate)
                if d < best_dist:
                    best_dist = d
                    best_idx = idx

            next_loc = unvisited.pop(best_idx)
            tour.append(next_loc)
            current = next_loc

        tour.append(depot)
        return tour


class SShapeBaseline(BaseRoutingHeuristic):
    """
    Heuristic lộ trình chữ S (Serpentine / S-Shape Routing Heuristic).

    Phương pháp dẫn đường kinh điển trong kho lưu trữ:
    - Gom nhóm các điểm lấy hàng theo từng dãy kệ (Aisle).
    - Duyệt qua các dãy kệ có điểm cần nhặt theo chiều ziczac luân phiên:
      * Dãy đầu tiên: Đi từ đầu dãy (Front cross-aisle) đến cuối dãy (Back cross-aisle).
      * Dãy tiếp theo: Đi từ cuối dãy trở ngược lại đầu dãy.
    - Cuối cùng quay trở về vị trí Depot.
    """

    def __init__(self) -> None:
        super().__init__(
            name="s_shape",
            baseline_type="reconstructed_baseline",
            description="Heuristic tái dựng: Lộ trình chữ S (Serpentine / S-Shape ziczac theo dãy kệ)",
        )

    def route(
        self,
        depot: Location,
        pick_locations: List[Location],
        distance_fn: Optional[Callable[[Location, Location], float]] = None,
    ) -> List[Location]:
        if not pick_locations:
            return [depot, depot]

        # 1. Gom nhóm vị trí theo Aisle (hoặc tọa độ x nếu không có aisle tường minh)
        aisle_groups: Dict[Any, List[Location]] = {}
        for loc in pick_locations:
            key = loc.aisle if loc.aisle is not None else round(loc.x, 1)
            aisle_groups.setdefault(key, []).append(loc)

        # 2. Sắp xếp các dãy kệ theo khoảng cách x (từ gần depot nhất đến xa nhất)
        sorted_aisle_keys = sorted(
            aisle_groups.keys(),
            key=lambda k: min(loc.x for loc in aisle_groups[k]),
        )

        tour = [depot]
        # 3. Duyệt luân phiên (Chẵn: Y tăng dần; Lẻ: Y giảm dần)
        for aisle_idx, key in enumerate(sorted_aisle_keys):
            locs_in_aisle = aisle_groups[key]
            reverse_direction = (aisle_idx % 2 == 1)
            sorted_locs = sorted(locs_in_aisle, key=lambda loc: loc.y, reverse=reverse_direction)
            tour.extend(sorted_locs)

        tour.append(depot)
        return tour


# ==============================================================================
# Factory & Evaluation Functions
# ==============================================================================

AVAILABLE_BASELINES: Dict[str, Callable[[bool], BaseRoutingHeuristic]] = {
    "original": lambda is_real: OriginalCompanyRouteBaseline(is_real_data=is_real),
    "company": lambda is_real: OriginalCompanyRouteBaseline(is_real_data=is_real),
    "actual": lambda is_real: OriginalCompanyRouteBaseline(is_real_data=is_real),
    "fifo": lambda is_real: ReconstructedFIFOBaseline(),
    "nearest_neighbor": lambda is_real: NearestNeighborBaseline(),
    "nn": lambda is_real: NearestNeighborBaseline(),
    "s_shape": lambda is_real: SShapeBaseline(),
    "sshape": lambda is_real: SShapeBaseline(),
    "serpentine": lambda is_real: SShapeBaseline(),
}


def get_baseline_heuristic(
    name: str = "original",
    is_real_data: bool = True,
) -> BaseRoutingHeuristic:
    """
    Khởi tạo đối tượng heuristic cơ sở theo tên đăng ký.

    Args:
        name: Tên thuật toán ('original', 'fifo', 'nearest_neighbor', 's_shape').
        is_real_data: True nếu chạy trên dữ liệu kho thực tế, False nếu mô phỏng.

    Returns:
        Đối tượng kế thừa BaseRoutingHeuristic.
    """
    key = name.lower().strip().replace("-", "_")
    if key not in AVAILABLE_BASELINES:
        raise ValueError(
            f"Không tìm thấy heuristic '{name}'. Các lựa chọn khả dụng: {list(AVAILABLE_BASELINES.keys())}"
        )
    return AVAILABLE_BASELINES[key](is_real_data)


def evaluate_baseline(
    wave: PickingWave,
    warehouse: WarehouseLayout,
    method: str = "original",
    is_real_data: bool = True,
    storage_policy: Optional[StoragePolicy] = None,
    distance_fn: Optional[Callable[[Location, Location], float]] = None,
) -> BaselineResult:
    """
    Đánh giá cự ly và tạo lộ trình cơ sở (Baseline) cho một PickingWave.

    Args:
        wave: Đợt nhặt hàng cần đánh giá.
        warehouse: Mô hình kho hàng (WarehouseLayout).
        method: Tên thuật toán cơ sở ('original', 'nearest_neighbor', 's_shape', 'fifo').
        is_real_data: Phân loại dữ liệu thực tế hay mô phỏng.
        storage_policy: Chính sách lưu trữ tùy chọn (mặc định lấy theo vị trí gốc của wave).
        distance_fn: Hàm tính khoảng cách tùy biến (mặc định Euclid 3D).

    Returns:
        Đối tượng BaselineResult ghi nhận đầy đủ cự ly, danh sách trạm dừng và phân loại.
    """
    start_time = time.perf_counter()
    heuristic = get_baseline_heuristic(name=method, is_real_data=is_real_data)

    depot, pick_locations = build_gro_route_targets(
        wave=wave,
        warehouse=warehouse,
        storage_policy=storage_policy,
    )

    dist_calc = distance_fn or (lambda a, b: a.distance_to(b))
    tour_locations = heuristic.route(depot, pick_locations, distance_fn=dist_calc)

    # Tính tổng cự ly
    total_dist = 0.0
    for i in range(len(tour_locations) - 1):
        total_dist += dist_calc(tour_locations[i], tour_locations[i + 1])

    elapsed = time.perf_counter() - start_time
    stop_ids = [loc.id for loc in tour_locations]
    num_picks = max(0, len(tour_locations) - 2) if len(tour_locations) >= 2 else 0

    return BaselineResult(
        wave_id=wave.wave_id,
        baseline_name=heuristic.name,
        baseline_type=heuristic.baseline_type,
        total_distance=round(total_dist, 3),
        stops=stop_ids,
        stops_count=num_picks,
        execution_time_seconds=elapsed,
        description=heuristic.description,
    )


def evaluate_all_baselines(
    wave: PickingWave,
    warehouse: WarehouseLayout,
    methods: Sequence[str] = ("original", "nearest_neighbor", "s_shape"),
    is_real_data: bool = True,
    storage_policy: Optional[StoragePolicy] = None,
    distance_fn: Optional[Callable[[Location, Location], float]] = None,
) -> Dict[str, BaselineResult]:
    """
    Đánh giá đồng thời tất cả các giải thuật cơ sở trên cùng một wave.

    Returns:
        Dict ánh xạ baseline_name -> BaselineResult.
    """
    results: Dict[str, BaselineResult] = {}
    for m in methods:
        res = evaluate_baseline(
            wave=wave,
            warehouse=warehouse,
            method=m,
            is_real_data=is_real_data,
            storage_policy=storage_policy,
            distance_fn=distance_fn,
        )
        results[res.baseline_name] = res
    return results
