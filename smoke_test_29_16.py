"""冒烟测试：29_16.mp4 PPT识别"""
from video_page_detector.streaming_pipeline import VideoPageDetector
from video_page_detector.config import DetectorConfig
import json, time, sys

config = DetectorConfig()
detector = VideoPageDetector(config)

page_order = []
def on_page(page, completed, total):
    page_order.append(page['page_id'])
    print(f"  [页面就绪] 第{page['page_id']}页 (已完成{completed}/{total})", flush=True)

print("开始PPT识别...")
result = detector.run(
    'test_vedio/29_16.mp4',
    output_root='output',
    video_id='smoke_test_29_16',
    progress_callback=lambda msg, prog: print(f"  [{prog:.0%}] {msg}", flush=True) if prog else None,
    page_ready_callback=on_page,
)

pages = result['pages']
analysis = result['analysis']
print(f"\n=== 冒烟测试结果 ===")
print(f"总页数: {len(pages)} (期望14)")
print(f"流式确认: {analysis.get('streaming_page_confirmation')}")
print(f"第一页交付时间: {analysis.get('first_page_ready_after_sec', 'N/A')}秒")
print(f"页面顺序: {page_order}")
for p in pages:
    print(f"  第{p['page_id']}页: {p['start_sec']:.1f}s-{p['end_sec']:.1f}s conf={p['confidence']}")

# 检查片头目录是否只有一页
# 片头目录通常在视频开头，如果前两页start_sec接近且都<10s，可能是有问题的
early_pages = [p for p in pages if p['start_sec'] < 30]
print(f"\n前30秒内页面数: {len(early_pages)}")
for p in early_pages:
    print(f"  第{p['page_id']}页: {p['start_sec']:.1f}s-{p['end_sec']:.1f}s")

# 验证
assert len(pages) == 14, f"页数错误: {len(pages)} != 14"
assert analysis.get('streaming_page_confirmation') == True, "非流式确认"
assert analysis.get('first_page_ready_after_sec') is not None, "第一页交付时间缺失"
assert page_order == list(range(1, len(pages)+1)), f"页面顺序异常: {page_order}"
# 片头目录应只有一页（第1页）
assert len(early_pages) <= 2, f"前30秒页面过多({len(early_pages)})，可能片头目录未合并"

print("\n[PASS] All smoke tests passed!")
