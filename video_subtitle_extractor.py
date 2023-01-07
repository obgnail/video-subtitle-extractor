import argparse
import cv2
import logging
import os

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta
from typing import List, Tuple, Callable, Iterable, Dict

from fuzzywuzzy import fuzz
from tqdm import tqdm

# ***** hardcode config *****
# 字幕最长显示秒数
subtitle_max_show_second = 10
# 字幕相似度阈值(大于此阈值判定为相似)
text_similar_threshold = 70


@contextmanager
def capture_video(video_path: str) -> Callable:
    vc = cv2.VideoCapture(video_path)
    if not vc.isOpened():
        raise IOError(f'Can not open video {video_path}')
    try:
        yield vc
    finally:
        vc.release()


# 接受一个帧索引迭代器,返回对应的每一帧画面
def get_video_frames(video_path: str, frame_idx_iterator: Iterable = None) -> Iterable:
    if frame_idx_iterator and (not isinstance(frame_idx_iterator, Iterable)):
        raise AttributeError("frame_idx_iterator must be Iterable")

    with capture_video(video_path) as vc:
        if frame_idx_iterator is None:
            idx = 0
            while True:
                ret, frame = vc.read()
                if ret == False or frame is None:
                    return
                yield idx, frame
                idx += 1
        else:
            for idx in frame_idx_iterator:
                vc.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = vc.read()
                if ret == False or frame is None:
                    return
                yield idx, frame


def convert_time_to_frame_idx(time_str: str, fps: int) -> int:
    if not time_str:
        return 0

    t = [float(i) for i in time_str.split(':')]
    if len(t) == 3:
        td = timedelta(hours=t[0], minutes=t[1], seconds=t[2])
    elif len(t) == 2:
        td = timedelta(minutes=t[0], seconds=t[1])
    else:
        raise ValueError(f'Time data "{time_str}" does not match format "%H:%M:%S"')
    index = int(td.total_seconds() * fps)
    return index


@dataclass(frozen=True)
class OcrResult:
    box: List[List[int]]
    text: str
    score: int


@dataclass(frozen=True)
class Subtitle(OcrResult):
    frame_idx: int

    def __le__(self, other):
        return self.frame_idx <= other.frame_idx

    def __str__(self):
        return f'({self.frame_idx}){self.text}'

    __repr__ = __str__


class SubtitleOption:
    """字幕操作"""
    threshold: int = text_similar_threshold

    @classmethod
    def order_by_frame(cls, subtitles: List[Subtitle]) -> List[List[Subtitle]]:
        keys: List[int] = []
        sub_dict: Dict[int, List[Subtitle]] = dict()
        group: List[List[Subtitle]] = []

        for sub in subtitles:
            if sub_dict.get(sub.frame_idx):
                sub_dict[sub.frame_idx].append(sub)
            else:
                sub_dict[sub.frame_idx] = [sub]
            keys.append(sub.frame_idx)

        for key in sorted(keys):
            group.append(sub_dict[key])

        return group

    @classmethod
    def removed_similar(cls, sub_order_by_frame: List[List[Subtitle]]) -> List[List[Subtitle]]:
        res = [sub_order_by_frame[0]]
        for subs in sub_order_by_frame[1:]:
            if not cls.subtitles_similar(subs, res[-1]):
                res.append(subs)
            else:
                res[-1] = cls.choose_better(subs, res[-1])
        return res

    @classmethod
    def clean(cls, subtitles: List[Subtitle]) -> List[List[Subtitle]]:
        if not subtitles:
            return []
        frame_subs = cls.order_by_frame(subtitles)
        frame_subs = cls.removed_similar(frame_subs)
        return frame_subs

    @classmethod
    def at_same_flame(cls, subs: List[Subtitle]) -> bool:
        if not subs:
            return True
        s = subs[0]
        for sub in subs[1:]:
            if s.frame_idx != sub.frame_idx:
                return False
        return True

    @classmethod
    def calc_avg_score(cls, subs: List[Subtitle]) -> float:
        """计算某一帧字幕的平均分"""
        score = 0
        for sub in subs:
            score += sub.score
        return score / len(subs)

    @classmethod
    def choose_better(cls, subs1: List[Subtitle], subs2: List[Subtitle]) -> List[Subtitle]:
        """选取某一帧平均分较高的字幕"""
        s1 = cls.calc_avg_score(subs1)
        s2 = cls.calc_avg_score(subs2)
        return subs1 if s1 > s2 else subs2

    @classmethod
    def subtitles_similar(cls, subs1: List[Subtitle], subs2: List[Subtitle]) -> bool:
        """两帧的字幕是否相似"""
        text_list1, text_list2 = [], []
        for sub in subs1:
            text_list1.append(sub.text)
        for sub in subs2:
            text_list2.append(sub.text)
        text1 = ' '.join(text_list1)
        text2 = ' '.join(text_list2)

        return fuzz.ratio(text1, text2) >= cls.threshold


@dataclass(frozen=True)
class SubtitleFormatter:
    content: str
    start_time: timedelta
    end_time: timedelta

    @property
    def lrc(self) -> str:
        return f'[{self.start_time}]{self.content}\n[{self.end_time}]'

    @property
    def txt(self) -> str:
        return f'{self.start_time} --> {self.end_time}\n{self.content}\n\n'


class Video:
    path: str
    frame_count: int
    fps: int
    height: int
    width: int

    def __init__(self, path):
        self.path = path
        with capture_video(path) as v:
            self.frame_count = int(v.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = round(v.get(cv2.CAP_PROP_FPS))
            self.height = int(v.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.width = int(v.get(cv2.CAP_PROP_FRAME_WIDTH))

    def time_to_frame_idx(self, time_str: str) -> int:
        """获取某一时刻对应的帧索引"""
        return convert_time_to_frame_idx(time_str, self.fps)

    def time_to_frame_idxes(self, time_start: str, time_end: str, capture_interval: float) -> Iterable:
        """获取某一时间范围对应的帧索引集合"""
        frame_start = 0 if time_start == '-' else self.time_to_frame_idx(time_start)
        frame_end = self.frame_count - 1 if time_end == '-' else self.time_to_frame_idx(time_end)
        if frame_end < frame_start:
            raise ValueError('time_start is later than time_end')
        step = int(capture_interval * self.fps)
        for frame_idx in range(frame_start, frame_end, step):
            yield frame_idx

    def count_frame(self, time_start: str, time_end: str, capture_interval: float) -> int:
        indexes = self.time_to_frame_idxes(time_start, time_end, capture_interval)
        count = 0
        for _ in indexes:
            count += 1
        return count

    def get_frames(self, frame_idx_iterator: Iterable = None) -> Iterable:
        """输入帧索引集合, 获取每一帧对应的画面"""
        return get_video_frames(self.path, frame_idx_iterator)

    def get_frames_by_time_range(self, time_start: str, time_end: str, capture_interval: float = 0.5) -> Iterable:
        """输入时间范围, 获取每一帧对应的画面"""
        indexes = self.time_to_frame_idxes(time_start, time_end, capture_interval)
        return self.get_frames(indexes)

    def get_frames_by_frame_range(self, frame_start: int, frame_end: int, frame_step: int) -> Iterable:
        """输入帧索引范围, 获取每一帧对应的画面"""
        frame_end = self.frame_count if not frame_end else frame_end + 1
        return self.get_frames(range(frame_start, frame_end, frame_step))

    def show_by_time_range(self,
                           frame_handler: Callable = None,
                           time_start: str = '-',
                           time_end: str = '-',
                           capture_interval: float = 0.5,
                           wait: int = 24) -> None:
        """输入时间范围, 展示对应的每一帧画面"""
        assert capture_interval > 0
        return self.show(
            frame_iterator=self.get_frames_by_time_range(time_start, time_end, capture_interval),
            frame_handler=frame_handler,
            wait=wait
        )

    def show_by_frame_range(self,
                            frame_handler: Callable = None,
                            frame_start: int = -1,
                            frame_end: int = -1,
                            frame_interval: int = 1,
                            wait: int = 24) -> None:
        """输入帧索引范围, 展示对应的每一帧画面"""
        assert frame_interval > 0
        frame_start = frame_start if frame_start != -1 else 0
        frame_end = frame_end if frame_end != -1 else self.frame_count - 1
        return self.show(
            frame_iterator=self.get_frames_by_frame_range(frame_start, frame_end, frame_interval),
            frame_handler=frame_handler,
            wait=wait
        )

    def show(self, frame_iterator: Iterable, frame_handler: Callable = None, wait: int = 24) -> None:
        """输入帧索引迭代器, 展示对应的每一帧画面"""
        assert wait > 0
        for idx, frame in frame_iterator:
            if frame_handler:
                frame = frame_handler(frame, self)
            cv2.imshow('Show frame. press ESC to Cancel, S to Save', frame)
            key = cv2.waitKey(wait)
            if key == 27:  # esc
                break
            elif key == ord('s'):
                cv2.imwrite(str(idx), frame)
        cv2.destroyAllWindows()

    def select_roi(self, frame_time: str = '-', resize: float = 0.5,
                   reshow: bool = False, reshow_wait: int = 24) -> Tuple[int]:
        """
        以交互的方式剪切某一时刻的画面
        :param frame_time: 时间
        :param resize: 如果完全展示2K、4K视频, 屏幕可能不够大, 提供缩放功能
        :param reshow: 剪切完毕后展示所选区域
        :return: tuple(矩形框中最小的x值, 矩形框中最小的y值, 矩形框的宽, 矩形框的高)
        """
        frame_index = 0 if frame_time == '-' else self.time_to_frame_idx(frame_time)
        with capture_video(self.path) as video:
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = video.read()
            if ret == False or frame is None:
                raise AttributeError(f'read frame error. start_time:{frame_time}')
            x, y = frame.shape[0:2]
            frame = cv2.resize(frame, (int(y * resize), int(x * resize)))
            roi = cv2.selectROI('Select a ROI. press SPACE or ENTER button to Confirm', frame, True, False)

        cv2.destroyAllWindows()

        r = tuple(int(i // resize) for i in roi)
        if reshow:
            def frame_handler(frame, video: Video):
                frame = FrameHandler.roi(frame, video, r)
                frame = FrameHandler.resize(frame, video, resize)
                return frame

            self.show_by_frame_range(
                frame_handler=frame_handler,
                frame_start=frame_index,
                frame_end=self.frame_count - 1,
                frame_interval=1,
                wait=reshow_wait
            )
        return r

    @staticmethod
    def _check_file_type(file_type: str) -> str:
        file_type = file_type.lower()
        file_type_list = tuple(name
                               for name, obj in vars(SubtitleFormatter).items()
                               if isinstance(obj, property))
        if file_type not in file_type_list:
            raise AttributeError(f'supported file type:{file_type_list}, got {file_type}')
        return file_type

    def save_subtitle(self, subtitles: str, file_type: str) -> None:
        basename = os.path.basename(self.path)
        file_name, file_ext = os.path.splitext(basename)
        file_path = f'{file_name}.{file_type}'
        with open(file_path, "w") as file:
            file.write(subtitles)

    def save_subtitle_by_formatter(self, formatters: List[SubtitleFormatter], file_type: str = 'lrc') -> None:
        suffix = self._check_file_type(file_type)
        subtitle_list = [getattr(formatter, suffix) for formatter in formatters]
        content = '\n'.join(subtitle_list)
        self.save_subtitle(content, file_type)


class FrameHandler:
    @classmethod
    def resize(cls, frame, video: Video, resize: float = 0.5):
        if resize != 1:
            x, y = frame.shape[0:2]
            frame = cv2.resize(frame, (int(y * resize), int(x * resize)))
        return frame

    @classmethod
    def gray(cls, frame, video: Video):
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    @classmethod
    def roi(cls, frame, video: Video, r: Tuple):
        return frame[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]


class SubtitleExtractor:
    video: Video
    frame_handler: Callable  # func(frame, video: Video) -> frame
    roi_array: Tuple[int]

    def __init__(self, video_path: str, *, roi_array: Tuple[int] = (), frame_handler: Callable = None):
        self.video = Video(path=video_path)
        self.frame_handler = frame_handler
        self.roi_array = roi_array
        self.table = str.maketrans('|', 'I', '<>{}[];`@#$%^*_=~\\')

    def ocr(self, ocr_handler, frame, frame_idx) -> List[Subtitle]:
        ocr_result = ocr_handler.ocr(frame, cls=True)
        if len(ocr_result) == 0:
            return []

        return [
            Subtitle(
                box=res[0],
                # remove chars that are obviously ocr errors
                text=res[1][0].translate(self.table).strip(),
                score=res[1][1],
                frame_idx=frame_idx
            )
            for res in ocr_result[0]
        ]

    def _to_formatter(self, subtitles: List[List[Subtitle]]) -> List[SubtitleFormatter]:
        if not subtitles:
            raise AttributeError('len(subtitles) == 0')

        last_sub_alive = subtitle_max_show_second * self.video.fps  # 最后一句字幕默认持续10s

        frame_idxes = [subs[0].frame_idx for subs in subtitles]
        frame_idxes.append(subtitles[-1][0].frame_idx + last_sub_alive)

        res = []
        for frame_idx in range(0, len(subtitles)):
            if not subtitles[frame_idx]:
                continue

            first_sub = subtitles[frame_idx][0]
            alive = frame_idxes[frame_idx + 1] - frame_idxes[frame_idx]  # 下一个字幕开始帧减去当前字幕开始帧
            start_second = first_sub.frame_idx // self.video.fps
            end_second = min(start_second + subtitle_max_show_second, (first_sub.frame_idx + alive) // self.video.fps)

            res.append(SubtitleFormatter(
                content=' '.join([sub.text for sub in subtitles[frame_idx]]),  # 将同一帧的字幕都合并起来
                start_time=timedelta(seconds=start_second),
                end_time=timedelta(seconds=end_second)
            ))
        return res

    def select_roi(self, time_start: str = '-', resize: float = 0.5,
                   reshow: bool = False, reshow_wait: int = 24) -> None:
        roi = self.video.select_roi(frame_time=time_start, resize=resize, reshow=reshow, reshow_wait=reshow_wait)
        self.roi_array = roi

    def extract_by_func(self, *,
                        ocr_handler,
                        frame_handler: Callable,
                        time_start: str = '-',
                        time_end: str = '-',
                        capture_interval: float = 0.5,
                        ) -> List[List[Subtitle]]:
        subtitles = []
        func = frame_handler or self.frame_handler
        for idx, frame in tqdm(
                iterable=self.video.get_frames_by_time_range(time_start, time_end, capture_interval),
                total=self.video.count_frame(time_start, time_end, capture_interval),
                unit='帧'):
            if func:
                frame = func(frame, self.video)
            if frame is not None:
                subtitle = self.ocr(ocr_handler, frame, idx)
                subtitles.extend(subtitle)
        subtitles = SubtitleOption.clean(subtitles)
        return subtitles

    def extract(
            self, *,
            # ocr config
            lang: str = 'ch',
            use_angle_cls: bool = True,
            use_gpu: bool = True,
            drop_score: float = 0.5,
            # video config
            time_start: str = '-',
            time_end: str = '-',
            capture_interval: float = 0.5,
            # handle frame config
            gray: bool = False,
            resize: float = 1,
    ) -> List[List[Subtitle]]:
        def frame_handler(frame, video: Video):
            if (r := self.roi_array, r):
                frame = FrameHandler.roi(frame, video, r)
            if gray == True:
                frame = FrameHandler.gray(frame, video)
            if resize != 1:
                frame = FrameHandler.resize(frame, video, resize)
            return frame

        from paddleocr import PaddleOCR, paddleocr  # 因为PaddleOCR需要加载大量数据到内存中，延迟导入
        ocr_handler = PaddleOCR(lang=lang, use_angle_cls=use_angle_cls, use_gpu=use_gpu, drop_score=drop_score)
        paddleocr.logging.disable(logging.DEBUG)
        subtitles = self.extract_by_func(
            ocr_handler=ocr_handler,
            frame_handler=frame_handler,
            time_start=time_start,
            time_end=time_end,
            capture_interval=capture_interval
        )
        return subtitles

    def save(self, subtitles: List[List[Subtitle]], file_type: str = 'lrc') -> None:
        formatters = self._to_formatter(subtitles)
        self.video.save_subtitle_by_formatter(formatters, file_type)


def cmd_run() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument('--path', type=str, help='video path')

    parser.add_argument('--ocr_lang', type=str, default='ch', help='ocr language')
    parser.add_argument('--ocr_use_angle_cls', type=bool, default=True, help='ocr use angle cls')
    parser.add_argument('--ocr_use_gpu', type=bool, default=True, help='ocr use gpu')
    parser.add_argument('--ocr_drop_score', type=float, default=0.5, help='ocr drop score')

    parser.add_argument('--parse_start_time', type=str, default='-', help='parse start time. format: %H:%M:%S"')
    parser.add_argument('--parse_end_time', type=str, default='-', help='parse end time. format: %H:%M:%S')
    parser.add_argument('--parse_capture_interval', type=float, default=0.5, help='parse capture interval')
    parser.add_argument('--parse_gray', type=bool, default=False, help='parse gray frame')
    parser.add_argument('--parse_resize', type=float, default=1, help='parse resize frame')

    parser.add_argument('--roi_start_time', type=str, help='select roi time. format: %H:%M:%S"')
    parser.add_argument('--roi_resize', type=float, default=0.5, help='select roi resize')
    parser.add_argument('--roi_reshow', type=bool, default=False, help='reshow roi selected frame')

    parser.add_argument('--output_format', type=str, default='lrc', help='subtitle file format')

    args = parser.parse_args()

    if not args.path:
        raise AttributeError("arg 'path' is null")
    if not args.roi_start_time:
        raise AttributeError("arg 'roi_start_time' is null")

    extractor = SubtitleExtractor(video_path=args.path)
    extractor.select_roi(time_start=args.roi_start_time, resize=args.roi_resize, reshow=args.roi_reshow)
    subtitles = extractor.extract(
        lang=args.ocr_lang,
        use_angle_cls=args.ocr_use_angle_cls,
        use_gpu=args.ocr_use_gpu,
        drop_score=args.ocr_drop_score,
        time_start=args.parse_start_time,
        time_end=args.parse_end_time,
        capture_interval=args.parse_capture_interval,
        gray=args.parse_gray,
        resize=args.parse_resize,
    )
    extractor.save(subtitles, file_type=args.output_format)


if __name__ == '__main__':
    path = r'd:\myshare\anime\Cyberpunk Edgerunners\[orion origin] Cyberpunk Edgerunners [01] [1080p] [H265 AAC] [CHS] [ENG＆JPN stidio].mkv'
    extractor = SubtitleExtractor(video_path=path)
    extractor.select_roi(time_start='3:24', resize=0.5, reshow=True, reshow_wait=1000)
    # subtitles = extractor.extract(time_start='3:24', time_end='3:40', resize=1, gray=False)
    # extractor.save(subtitles, file_type='lrc')
