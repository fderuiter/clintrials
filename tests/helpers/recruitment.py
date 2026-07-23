from clintrials.core.recruitment import QuadrilateralRecruitmentStream


class QuadrilateralRecruitmentStreamBuilder:
    def __init__(self) -> None:
        self._intrapatient_gap = 15.0
        self._initial_intensity = 1.0
        self._vertices = [(90, 1.0)]
        self._interpolate = True

    def with_intrapatient_gap(self, intrapatient_gap: float) -> "QuadrilateralRecruitmentStreamBuilder":
        self._intrapatient_gap = intrapatient_gap
        return self

    def with_initial_intensity(self, initial_intensity: float) -> "QuadrilateralRecruitmentStreamBuilder":
        self._initial_intensity = initial_intensity
        return self

    def with_vertices(self, vertices: list) -> "QuadrilateralRecruitmentStreamBuilder":
        self._vertices = vertices
        return self

    def with_interpolate(self, interpolate: bool) -> "QuadrilateralRecruitmentStreamBuilder":
        self._interpolate = interpolate
        return self

    def build(self) -> QuadrilateralRecruitmentStream:
        return QuadrilateralRecruitmentStream(
            intrapatient_gap=self._intrapatient_gap,
            initial_intensity=self._initial_intensity,
            vertices=self._vertices,
            interpolate=self._interpolate
        )
