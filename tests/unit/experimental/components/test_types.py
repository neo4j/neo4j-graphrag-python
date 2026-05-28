#  Copyright (c) "Neo4j"
#  Neo4j Sweden AB [https://neo4j.com]
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#      https://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from __future__ import annotations

import pytest


def test_pdf_document_alias_is_loaded_document_and_warns() -> None:
    """Accessing PdfDocument emits DeprecationWarning and returns LoadedDocument."""
    import neo4j_graphrag.experimental.components.types as types_mod

    with pytest.warns(DeprecationWarning, match="PdfDocument is deprecated"):
        PdfDocument = types_mod.PdfDocument

    from neo4j_graphrag.experimental.components.types import LoadedDocument

    assert PdfDocument is LoadedDocument


def test_pdf_document_from_import_backward_compat_warns() -> None:
    """``from ...types import PdfDocument`` still works and emits DeprecationWarning."""
    with pytest.warns(DeprecationWarning, match="PdfDocument is deprecated"):
        from neo4j_graphrag.experimental.components.types import (
            LoadedDocument,
            PdfDocument,
        )

    assert PdfDocument is LoadedDocument


def test_types_dir_includes_pdf_document() -> None:
    import neo4j_graphrag.experimental.components.types as types_mod

    assert "PdfDocument" in dir(types_mod)
    assert "LoadedDocument" in dir(types_mod)


def test_getattr_unknown_raises() -> None:
    import neo4j_graphrag.experimental.components.types as types_mod

    with pytest.raises(AttributeError, match="no attribute 'not_a_real_export'"):
        _ = types_mod.not_a_real_export
