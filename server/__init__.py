
#server/__init__.py
# Copyright 2020 Adap GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""FedOps Server."""

# from .app import FLServer
# from .server_api import ServerAPI
# from .server_utils import FLServerStatus

from . import app as app
from . import server_utils as server_utils
from . import server_api as server_api
from . import mobile_app as mobile_app
from . import mobile_strategy as mobile_strategy


__all__ = [
    "app",
    "server_utils",
    "server_api",
    "mobile_app",
    "mobile_strategy",
]
