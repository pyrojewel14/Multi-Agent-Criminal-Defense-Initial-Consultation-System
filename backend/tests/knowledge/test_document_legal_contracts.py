"""保留法律文档结构识别与 metadata 提取的既有边界。"""

import json

from langchain_core.documents import Document

from app.knowledge.documents.splitters.legal import LegalArticleSplitter


class TestIsStructuredLegalDoc:
    def setup_method(self):
        self.splitter = LegalArticleSplitter()

    def test_structured_json_list_with_law_text(self):
        content = json.dumps([{"law_text": "第一条 ...", "article": "第一条"}])
        doc = Document(page_content=content, metadata={})
        assert self.splitter._is_structured_legal_doc(doc) is True

    def test_structured_json_dict_with_law_text(self):
        content = json.dumps({"law_text": "第一条 ...", "article": "第一条"})
        doc = Document(page_content=content, metadata={})
        assert self.splitter._is_structured_legal_doc(doc) is True

    def test_source_type_legal_json(self):
        doc = Document(page_content="anything", metadata={"source_type": "legal_json"})
        assert self.splitter._is_structured_legal_doc(doc) is True

    def test_plain_text_not_structured(self):
        doc = Document(page_content="这是一段普通法律文本，没有JSON结构。", metadata={})
        assert self.splitter._is_structured_legal_doc(doc) is False

    def test_json_without_law_text_not_structured(self):
        content = json.dumps([{"title": "标题", "body": "内容"}])
        doc = Document(page_content=content, metadata={})
        assert self.splitter._is_structured_legal_doc(doc) is False

    def test_invalid_json_not_structured(self):
        doc = Document(page_content="{invalid json", metadata={})
        assert self.splitter._is_structured_legal_doc(doc) is False


class TestExtractLegalMetadata:
    def setup_method(self):
        self.splitter = LegalArticleSplitter(enable_metadata_extraction=True)

    def test_extract_article_index(self):
        text = "根据刑法第232条的规定……"
        metadata = {}
        result = self.splitter._extract_legal_metadata(text, metadata)
        assert result["article_index"] == 232

    def test_extract_charge_from_parentheses(self):
        text = "故意杀人罪（故意杀人）处死刑……"
        metadata = {}
        result = self.splitter._extract_legal_metadata(text, metadata)
        assert result["charge"] == "故意杀人"

    def test_existing_charge_not_overwritten(self):
        text = "故意伤害罪（故意伤害）……"
        metadata = {"charge": "故意杀人罪"}
        result = self.splitter._extract_legal_metadata(text, metadata)
        assert result["charge"] == "故意杀人罪"

    def test_death_related_category(self):
        text = "致人死亡的，处十年以上有期徒刑。"
        metadata = {}
        result = self.splitter._extract_legal_metadata(text, metadata)
        assert result["legal_category"] == "death_related"

    def test_injury_related_category(self):
        text = "致人重伤的，处三年以上十年以下有期徒刑。"
        metadata = {}
        result = self.splitter._extract_legal_metadata(text, metadata)
        assert result["legal_category"] == "injury_related"

    def test_property_crime_category(self):
        text = "盗窃公私财物，数额较大的……"
        metadata = {}
        result = self.splitter._extract_legal_metadata(text, metadata)
        assert result["legal_category"] == "property_crime"

    def test_danger_related_category(self):
        text = "危害公共安全的……"
        metadata = {}
        result = self.splitter._extract_legal_metadata(text, metadata)
        assert result["legal_category"] == "danger_related"

    def test_general_category(self):
        text = "国家工作人员利用职务上的便利……"
        metadata = {}
        result = self.splitter._extract_legal_metadata(text, metadata)
        assert result["legal_category"] == "general"

    def test_mitigated_article_type(self):
        text = "情节较轻的，可以从轻处罚。"
        metadata = {}
        result = self.splitter._extract_legal_metadata(text, metadata)
        assert result["article_type"] == "mitigated"

    def test_default_article_type_is_standard(self):
        text = "国家工作人员利用职务上的便利……"
        metadata = {}
        result = self.splitter._extract_legal_metadata(text, metadata)
        assert result["article_type"] == "standard"
