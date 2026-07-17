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
"""Backward-compatibility tests for the moved ``components`` namespace.

Components moved from ``neo4j_graphrag.experimental.components`` to
``neo4j_graphrag.components``; the old import paths must keep working (with a
``DeprecationWarning``) via the meta path finder in
``neo4j_graphrag/experimental/components/__init__.py``.
"""

import importlib
import warnings

import pytest


def test_old_import_path_redirects_to_new_module() -> None:
    with pytest.warns(DeprecationWarning, match="has moved to"):
        from neo4j_graphrag.experimental.components.types import (
            LexicalGraphConfig as OldLexicalGraphConfig,
        )

    from neo4j_graphrag.components.types import LexicalGraphConfig

    # Same object under both paths: not a copy, identity/isinstance hold.
    assert OldLexicalGraphConfig is LexicalGraphConfig


def test_nested_subpackage_redirects() -> None:
    with pytest.warns(DeprecationWarning, match="has moved to"):
        from neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter import (  # noqa: E501
            FixedSizeSplitter as OldSplitter,
        )

    from neo4j_graphrag.components.text_splitters.fixed_size_splitter import (
        FixedSizeSplitter,
    )

    assert OldSplitter is FixedSizeSplitter


def test_import_module_form_shares_the_same_module() -> None:
    with pytest.warns(DeprecationWarning, match="has moved to"):
        old = importlib.import_module(
            "neo4j_graphrag.experimental.components.kg_writer"
        )
    new = importlib.import_module("neo4j_graphrag.components.kg_writer")

    assert old is new
