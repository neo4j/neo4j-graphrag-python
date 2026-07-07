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
import warnings
import importlib
from typing import Any


def __getattr__(name: str) -> Any:
    warnings.warn(
        f"Importing component {name} from the experimental namespace is deprecated and will be removed in version `2.0`. Use `from neo4j_graphrag.components... import ...` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    try:
        print(name)
        module_split = name.rsplit(".", 1)
        module_name = "neo4j_graphrag.components"
        if len(module_split) == 1:
            klass_name = module_split[0]
        else:
            module_name = module_name + "." + module_split[0]
            klass_name = module_split[1]
        print(module_split, module_name, klass_name)
        module = importlib.import_module(module_name)
        klass = getattr(module, klass_name)
        return klass
    except (ImportError, AttributeError) as e:
        print(e)
        raise
