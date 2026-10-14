import json
import re
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document


class LegalArticleSplitter:
    """
    刑法专用文本分割器

    核心功能：
    1. 识别并保留法条作为最小语义单元
    2. 支持款、项级别的细粒度分割
    3. 自动提取法条编号和罪名信息
    4. 处理法律文书特有的引用关系
    """

    def __init__(
        self,
        chunk_size: int = 2000,
        chunk_overlap: int = 200,
        preserve_structure: bool = True,
        enable_metadata_extraction: bool = True,
    ):
        """
        初始化法律文本分割器

        Args:
            chunk_size: 每个文本片段的最大长度
            chunk_overlap: 片段之间的重叠长度
            preserve_structure: 是否保留法律文档结构
            enable_metadata_extraction: 是否启用元数据提取
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.preserve_structure = preserve_structure
        self.enable_metadata_extraction = enable_metadata_extraction

        self.article_pattern = re.compile(r"第[一二三四五六七八九十百零\d]+条")
        self.chapter_pattern = re.compile(r"第[一二三四五六七八九十百零\d]+[编章]")
        self.paragraph_pattern = re.compile(r"第[一二三四五六七八九十百零\d]+款")

    def split_documents(self, documents: List[Any]) -> List[Document]:
        """
        分割文档，保留法条完整性

        Args:
            documents: 文档对象列表

        Returns:
            List[Document]: 分割后的文档对象列表
        """
        split_docs = []

        for doc in documents:
            if self._is_structured_legal_doc(doc):
                split_docs.extend(self._split_structured_doc(doc))
            elif self.article_pattern.search(doc.page_content):
                split_docs.extend(self._split_by_articles(doc))
            else:
                split_docs.extend(self._split_by_paragraphs(doc))

        return split_docs

    def split_documents_sync(self, documents: List[Any]) -> List[Document]:
        """
        同步分割文档列表（用于多线程场景）

        Args:
            documents: 文档对象列表

        Returns:
            List[Document]: 分割后的文档对象列表
        """
        return self.split_documents(documents)

    def split_text(self, text: str) -> List[str]:
        """
        分割文本成多个片段

        Args:
            text: 要分割的文本

        Returns:
            List[str]: 分割后的文本片段列表
        """
        temp_doc = Document(page_content=text, metadata={})
        split_docs = self.split_documents([temp_doc])
        return [doc.page_content for doc in split_docs]

    def split_text_sync(self, text: str) -> List[str]:
        """
        同步分割文本（用于多线程场景）

        Args:
            text: 要分割的文本

        Returns:
            List[str]: 分割后的文本片段列表
        """
        return self.split_text(text)

    def _split_by_articles(self, doc: Document) -> List[Document]:
        """
        按法条分割文档（推荐方案）

        Args:
            doc: 文档对象

        Returns:
            List[Document]: 分割后的文档列表
        """
        content = doc.page_content
        matches = list(self.article_pattern.finditer(content))

        if not matches:
            return [doc]

        chunks = []

        for i, match in enumerate(matches):
            article_start = match.start()
            article_num = match.group(0)

            if i + 1 < len(matches):
                article_end = matches[i + 1].start()
            else:
                article_end = len(content)

            article_text = content[article_start:article_end].strip()

            if not article_text:
                continue

            metadata = doc.metadata.copy()
            metadata["article_number"] = article_num
            metadata["chunk_type"] = "article"
            metadata["article_text"] = article_text

            if self.enable_metadata_extraction:
                metadata = self._extract_legal_metadata(article_text, metadata)

            if len(article_text) > self.chunk_size:
                sub_chunks = self._split_long_article(article_text, article_num, metadata)
                chunks.extend(sub_chunks)
            else:
                chunks.append(Document(page_content=article_text, metadata=metadata))

        return chunks

    def _split_long_article(self, text: str, article_num: str, base_metadata: dict) -> List[Document]:
        """
        处理超长法条（按款分割）

        Args:
            text: 法条文本
            article_num: 法条编号
            base_metadata: 基础元数据

        Returns:
            List[Document]: 分割后的文档列表
        """
        sentences = re.split(r"([。；])", text)

        combined_sentences = []
        current = ""
        for i in range(0, len(sentences) - 1, 2):
            sentence = sentences[i] + sentences[i + 1]
            if len(current) + len(sentence) < self.chunk_size:
                current += sentence
            else:
                if current:
                    combined_sentences.append(current)
                current = sentence

        if current:
            combined_sentences.append(current)

        chunks = []
        for idx, chunk_text in enumerate(combined_sentences):
            metadata = base_metadata.copy()
            metadata["chunk_type"] = "paragraph"
            metadata["paragraph_index"] = idx
            metadata["is_partial_article"] = True

            chunks.append(Document(page_content=chunk_text.strip(), metadata=metadata))

        return chunks

    def _split_structured_doc(self, doc: Document) -> List[Document]:
        """
        分割结构化法律文档（JSON格式）

        Args:
            doc: 文档对象

        Returns:
            List[Document]: 分割后的文档列表
        """
        try:
            data = json.loads(doc.page_content)
            chunks = []

            if isinstance(data, list):
                for item in data:
                    chunk = self._create_chunk_from_item(item, doc.metadata)
                    if chunk:
                        chunks.append(chunk)
            elif isinstance(data, dict):
                chunk = self._create_chunk_from_item(data, doc.metadata)
                if chunk:
                    chunks.append(chunk)

            if not chunks:
                return [doc]

            return chunks
        except (json.JSONDecodeError, TypeError):
            return self._split_by_articles(doc)

    def _create_chunk_from_item(self, item: Dict[str, Any], base_metadata: dict) -> Optional[Document]:
        """
        从结构化数据项创建文档块

        Args:
            item: 数据项字典
            base_metadata: 基础元数据

        Returns:
            Optional[Document]: 文档对象，如果数据无效则返回None
        """
        if not isinstance(item, dict):
            return None

        if "law_text" in item:
            return self._create_chunk_from_law_text_format(item, base_metadata)
        elif "content" in item:
            return self._create_chunk_from_content_format(item, base_metadata)
        else:
            return None

    def _create_chunk_from_law_text_format(self, item: Dict[str, Any], base_metadata: dict) -> Optional[Document]:
        """
        从 law_text 格式创建文档块（原示例格式）

        Args:
            item: 数据项字典
            base_metadata: 基础元数据

        Returns:
            Optional[Document]: 文档对象
        """
        law_text = item.get("law_text", "")
        if not law_text or not law_text.strip():
            return None

        article_num = item.get("article", "")
        charge = item.get("charge", "")

        metadata = base_metadata.copy()
        metadata.update(
            {
                "article_number": article_num,
                "charge": charge,
                "chunk_type": "legal_unit",
                "elements": item.get("elements", []),
                "keywords": item.get("keywords", []),
                "base_sentence": item.get("base_sentence", ""),
                "law_text": law_text,
            }
        )

        if self.enable_metadata_extraction:
            metadata = self._extract_legal_metadata(law_text, metadata)

        return Document(page_content=law_text, metadata=metadata)

    def _create_chunk_from_content_format(self, item: Dict[str, Any], base_metadata: dict) -> Optional[Document]:
        """
        从 content 格式创建文档块（新刑法JSON格式）

        Args:
            item: 数据项字典
            base_metadata: 基础元数据

        Returns:
            Optional[Document]: 文档对象
        """
        content = item.get("content", "")
        if not content or not content.strip():
            return None

        title = item.get("title", "")
        article_info = self._extract_article_from_content(content)

        metadata = base_metadata.copy()
        metadata.update(
            {
                "title": title,
                "article_number": article_info["article_number"],
                "article_text": article_info["article_text"],
                "chunk_type": "legal_article",
                "source_format": "content_format",
            }
        )

        if self.enable_metadata_extraction:
            metadata = self._extract_legal_metadata(content, metadata)

        return Document(page_content=content, metadata=metadata)

    def _extract_article_from_content(self, content: str) -> dict:
        """
        从内容中提取法条信息

        Args:
            content: 内容文本

        Returns:
            dict: 包含 article_number 和 article_text 的字典
        """
        match = self.article_pattern.search(content)
        if match:
            return {"article_number": match.group(0), "article_text": content}

        first_line = content.split("\n")[0].strip()
        return {"article_number": first_line[:20] if len(first_line) > 20 else first_line, "article_text": content}

    def _split_by_paragraphs(self, doc: Document) -> List[Document]:
        """
        回退方案：按段落分割

        Args:
            doc: 文档对象

        Returns:
            List[Document]: 分割后的文档列表
        """
        content = doc.page_content

        paragraphs = re.split(r"\n\s*\n", content)

        chunks = []
        current_chunk = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if len(current_chunk) + len(para) < self.chunk_size:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
            else:
                if current_chunk:
                    metadata = doc.metadata.copy()
                    metadata["chunk_type"] = "paragraph"
                    chunks.append(Document(page_content=current_chunk, metadata=metadata))
                current_chunk = para

        if current_chunk:
            metadata = doc.metadata.copy()
            metadata["chunk_type"] = "paragraph"
            chunks.append(Document(page_content=current_chunk, metadata=metadata))

        if not chunks:
            return [doc]

        return chunks

    def _is_structured_legal_doc(self, doc: Document) -> bool:
        """
        检测是否为结构化法律文档

        Args:
            doc: 文档对象

        Returns:
            bool: 是否为结构化文档
        """
        source_type = doc.metadata.get("source_type", "")
        if source_type == "legal_json":
            return True

        content = doc.page_content.strip()
        if content.startswith("[") or content.startswith("{"):
            try:
                data = json.loads(content)
                if isinstance(data, list) and len(data) > 0:
                    first_item = data[0]
                    if isinstance(first_item, dict) and "law_text" in first_item:
                        return True
                elif isinstance(data, dict) and "law_text" in data:
                    return True
            except json.JSONDecodeError:
                pass

        return False

    def _extract_legal_metadata(self, text: str, metadata: dict) -> dict:
        """
        提取法律文档元数据

        Args:
            text: 法律文本
            metadata: 已有元数据

        Returns:
            dict: 更新后的元数据
        """
        article_match = re.search(r"刑法第([\d]+)条", text)
        if article_match:
            metadata["article_index"] = int(article_match.group(1))

        charge_match = re.search(r"（([^\)]+?)）", text)
        if charge_match and not metadata.get("charge"):
            metadata["charge"] = charge_match.group(1)

        sentence_patterns = [
            (r"处(三年以下有期徒刑|三年以上(?:十年以下|[^年]+)有期徒刑|十年以上有期徒刑|无期徒刑|死刑)", "standard"),
            (r"致人重伤", "aggravated"),
            (r"致人死亡", "most_severe"),
            (r"情节较轻", "mitigated"),
            (r"可以从轻", "mitigated"),
            (r"应当从轻", "mitigated"),
            (r"以特别残忍手段", "aggravated"),
        ]

        for pattern, article_type in sentence_patterns:
            if re.search(pattern, text):
                metadata["article_type"] = article_type
                break

        if "article_type" not in metadata:
            metadata["article_type"] = "standard"

        has_injury = any(kw in text for kw in ["伤害", "损伤", "轻伤", "重伤"])
        has_death = any(kw in text for kw in ["死亡", "致死"])
        has_property = any(kw in text for kw in ["盗窃", "抢劫", "诈骗", "侵占", "财产"])
        has_danger = any(kw in text for kw in ["危险", "危害", "公共安全"])

        if has_death:
            metadata["legal_category"] = "death_related"
        elif has_injury:
            metadata["legal_category"] = "injury_related"
        elif has_property:
            metadata["legal_category"] = "property_crime"
        elif has_danger:
            metadata["legal_category"] = "danger_related"
        else:
            metadata["legal_category"] = "general"

        return metadata
