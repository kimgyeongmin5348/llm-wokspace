from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Iterable


def normalize_unicode(text: str) -> str:
    """유니코드와 보이지 않는 문자를 정리한다."""
    if text is None:
        return ""

    text = unicodedata.normalize("NFKC", str(text))
    text = text.replace("\ufeff", "")
    text = text.replace("\u00a0", " ")
    text = text.replace("\u200b", "")
    text = text.replace("\u200c", "")
    text = text.replace("\u200d", "")
    text = text.replace("\x00", " ")

    return text


def remove_dot_leaders(text: str) -> str:
    """
    PDF 변환 과정에서 생기는 점선성 문자와 말줄임표를 정리한다.

    예:
    - "제목 ........ 3" → "제목 3"
    - "...." → 제거
    - "……" → 제거
    """
    text = re.sub(r"\.{3,}", " ", text)
    text = re.sub(r"…{1,}", " ", text)
    text = re.sub(r"·{4,}", " ", text)

    return text


def is_noise_line(line: str, remove_page_numbers: bool = True) -> bool:
    """의미 없는 라인인지 판정한다."""
    stripped = line.strip()

    if not stripped:
        return False

    # 점, 대시, 밑줄 등으로만 구성된 라인
    if re.fullmatch(r"[\s.·…\-–—_=]{3,}", stripped):
        return True

    # 특수문자로만 구성된 라인
    if re.fullmatch(r"[^\w가-힣A-Za-z0-9]{3,}", stripped):
        return True

    if remove_page_numbers:
        # 단독 페이지 번호: 3, - 3 -, 3 / 20
        if re.fullmatch(r"[-–—]?\s*\d+\s*[-–—]?", stripped):
            return True
        if re.fullmatch(r"\d+\s*/\s*\d+", stripped):
            return True

    return False


def clean_line(line: str) -> str:
    """라인 내부의 공백과 변환 잔여 문자를 정리한다."""
    line = normalize_unicode(line)
    line = remove_dot_leaders(line)

    # Markdown 이미지 placeholder 제거
    line = re.sub(r"!\[[^\]]*\]\([^)]+\)", " ", line)

    # 빈 링크 표현 정리
    line = re.sub(r"\[\s*\]\([^)]+\)", " ", line)

    # 여러 공백 정리
    line = re.sub(r"[ \t]+", " ", line)

    return line.strip()


def collapse_blank_lines(text: str, max_blank_lines: int = 1) -> str:
    """
    연속 공백 라인을 제한한다.

    max_blank_lines=1이면 문단 사이 빈 줄을 1개까지만 허용한다.
    """
    lines = text.splitlines()
    new_lines: list[str] = []
    blank_count = 0

    for line in lines:
        if line.strip() == "":
            blank_count += 1
            if blank_count <= max_blank_lines:
                new_lines.append("")
        else:
            blank_count = 0
            new_lines.append(line)

    return "\n".join(new_lines).strip()


def clean_markdown(
    text: str,
    *,
    remove_page_numbers: bool = True,
    max_blank_lines: int = 1,
) -> str:
    """
    Markdown 변환 결과를 정제한다.

    처리 내용:
    1. 유니코드 정규화
    2. 보이지 않는 문자 제거
    3. 점선성 문자 제거
    4. 의미 없는 라인 제거
    5. 공백 라인 축소
    """
    text = normalize_unicode(text)

    cleaned_lines: list[str] = []

    for raw_line in text.splitlines():
        line = clean_line(raw_line)

        if line == "":
            cleaned_lines.append("")
            continue

        if is_noise_line(line, remove_page_numbers=remove_page_numbers):
            continue

        cleaned_lines.append(line)

    cleaned_text = "\n".join(cleaned_lines)
    cleaned_text = collapse_blank_lines(cleaned_text, max_blank_lines=max_blank_lines)

    return cleaned_text


def find_repeated_lines(
    page_texts: Iterable[str],
    *,
    min_count: int = 4,
    max_len: int = 80,
) -> set[str]:
    """
    여러 페이지에 반복 출현하는 짧은 라인을 찾는다.

    주의:
    반복 라인 제거는 헤더/푸터 제거에는 유용하지만,
    문서의 공통 소제목까지 제거할 수 있으므로 신중하게 사용한다.
    """
    counter: Counter[str] = Counter()

    for page_text in page_texts:
        unique_lines = {
            clean_line(line)
            for line in page_text.splitlines()
            if clean_line(line)
        }

        for line in unique_lines:
            if len(line) <= max_len:
                # 표 라인과 Markdown 제목은 보존한다.
                if line.startswith("#"):
                    continue
                if "|" in line:
                    continue
                counter[line] += 1

    return {
        line
        for line, count in counter.items()
        if count >= min_count
    }


def remove_repeated_lines(text: str, repeated_lines: set[str]) -> str:
    """반복 라인 후보를 제거한다."""
    if not repeated_lines:
        return text

    kept_lines = []

    for line in text.splitlines():
        if clean_line(line) in repeated_lines:
            continue
        kept_lines.append(line)

    return "\n".join(kept_lines)


def clean_pages(
    page_texts: Iterable[str],
    *,
    remove_page_numbers: bool = True,
    remove_repeated: bool = False,
    repeated_min_count: int = 4,
    repeated_max_len: int = 80,
    max_blank_lines: int = 1,
) -> list[str]:
    """
    페이지 단위 텍스트 목록을 일괄 정제한다.

    기본값은 점선성 문자와 공백 라인을 제거하는 안전한 정제이다.
    remove_repeated=True로 설정하면 반복 헤더/푸터 후보도 제거한다.
    """
    page_texts = list(page_texts)

    first_pass = [
        clean_markdown(
            page_text,
            remove_page_numbers=remove_page_numbers,
            max_blank_lines=max_blank_lines,
        )
        for page_text in page_texts
    ]

    if not remove_repeated:
        return first_pass

    repeated_lines = find_repeated_lines(
        first_pass,
        min_count=repeated_min_count,
        max_len=repeated_max_len,
    )

    second_pass = [
        clean_markdown(
            remove_repeated_lines(page_text, repeated_lines),
            remove_page_numbers=remove_page_numbers,
            max_blank_lines=max_blank_lines,
        )
        for page_text in first_pass
    ]

    return second_pass


def cleaning_report(before: str, after: str) -> dict:
    """전처리 전후의 간단한 품질 지표를 반환한다."""
    before_blank_lines = sum(1 for line in before.splitlines() if line.strip() == "")
    after_blank_lines = sum(1 for line in after.splitlines() if line.strip() == "")

    before_dot_sequences = len(re.findall(r"\.{3,}|…+", before))
    after_dot_sequences = len(re.findall(r"\.{3,}|…+", after))

    return {
        "before_chars": len(before),
        "after_chars": len(after),
        "removed_chars": len(before) - len(after),
        "before_blank_lines": before_blank_lines,
        "after_blank_lines": after_blank_lines,
        "before_dot_sequences": before_dot_sequences,
        "after_dot_sequences": after_dot_sequences,
    }
