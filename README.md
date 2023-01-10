# video subtitle extractor

![video_subtitle_extractor](assets/video_subtitle_extractor.gif)

## introduction

提取视频硬字幕。采用 PaddleOCR。

在解析前可以对视频进行`节选`(select_fragment)，`框选范围`(select_roi)，`阈值`(select_threshold) 操作，提高 OCR 效率。

三者操作都可以使用 use_fragment，use_roi，use_threshold 配置进入交互页面设置。也可以通过 time_start、time_end、roi_array、threshold 等配置参数直接设置。

```bash
python video_subtitle_extractor.py --path="./CyberpunkEdgerunners01.mkv" --use_fragment=True --use_threshold=True --use_roi=True --r oi_time="3:24"
```

```python
extractor = SubtitleExtractor(
  video_path=r'./CyberpunkEdgerunners01.mkv'，
  time_start='03:01',
  time_end='23:44',
  roi_array=(24, 789, 1896 191),
  threshold=201,
)
# extractor.select_fragment()
# extractor.select_roi()
# extractor.select_threshold()
subtitles = extractor.extract()
extractor.save(subtitles)
```

## config

common：

- path：视频路径
- subtitle_max_show_second：字幕最长显示时间，默认 10s
- text_similar_threshold：字幕相似度阈值(大于此阈值判定为相似)，默认 70
- output_format：输出字幕的格式，默认 lrc

parser：

- parse_time_start：开始解析视频的时间。format: %H:%M:%S。
- parse_time_end：停止解析视频的时间。format: %H:%M:%S。
- parse_capture_interval：解析的采样率。默认每隔 0.5s 采样一次。
- parse_gray：解析视频时是否将视频切换为灰度图。默认 False。
- parse_resize：解析视频时是否将视频进行缩放处理。默认为 1。

prepare：

- use_fragment：是否通过交互页面设置 parse_time_start 和 parse_time_end。默认为 False（如果 parse_time_start 和 parse_time_end 都没有设置，use_fragment 将设置为 True）
- fragment_reshow：选取片段后，是否重新展示。
- use_roi：是否通过交互页面框选字幕位置。默认为 False
- roi_time：出现字幕的时间。
- roi_reshow：框选字幕位置后，是否重新展示。
- use_threshold：是否使用阈值对视频进行处理。
- threshold_time：出现字幕的时间。
- threshold_reshow：使用阈值后，是否重新展示。

ocr：详见 PaddleOCR 的配置

- ocr_lang
- ocr_use_angle_cls
- ocr_use_gpu
- ocr_use_mp
- ocr_enable_mkldnn
- ocr_gpu_mem
- ocr_det_limit_side_len
- ocr_rec_batch_num
- ocr_cpu_threads
- ocr_drop_score



