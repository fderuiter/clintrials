import pytest
from clintrials.core.viz_interface import VisualizationResult, get_visualization_provider, set_visualization_provider, _provider

class DummyChart:
    def _repr_html_(self):
        return "<div>chart</div>"

class DummyChartToHtml:
    def to_html(self, full_html=False, include_plotlyjs="cdn"):
        return "<div>chart_to_html</div>"

class DummyMetadata:
    def to_html(self, index=False):
        return "<table>metadata</table>"

def test_visualization_result_repr_html():
    res1 = VisualizationResult(chart=DummyChart(), metadata=DummyMetadata(), title="Test 1")
    html1 = res1._repr_html_()
    assert "<div>chart</div>" in html1
    assert "<table>metadata</table>" in html1
    assert "Accessibility Metadata: Test 1" in html1

    res2 = VisualizationResult(chart=DummyChartToHtml())
    html2 = res2._repr_html_()
    assert "<div>chart_to_html</div>" in html2
    assert "Accessibility Metadata" not in html2

    res3 = VisualizationResult(chart=None)
    assert res3._repr_html_() is None

def test_get_set_visualization_provider():
    import clintrials.core.viz_interface as vi
    # reset provider
    original_provider = vi._provider
    vi._provider = None
    
    # It should import default provider successfully since viz is installed
    provider = get_visualization_provider()
    assert provider is not None
    
    # test set
    set_visualization_provider("test")
    assert vi._provider == "test"
    
    # reset back
    vi._provider = original_provider
