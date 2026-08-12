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
"""Vertex AI, Bedrock and Azure OpenAI: the cloud logins.

Each vendor's CLI owns its own login flow, so this only offers to run it and
records the one value the examples read from the environment, the GCP project.
"""

from __future__ import annotations

from .console import BOLD, GREEN, colour, confirm, heading, run_command
from .envfile import write_env_var
from .probes import command_exists
from .providers import PROVIDERS


def cloud_logins(assume_yes: bool) -> None:
    heading("Cloud providers")
    print(
        "Vertex AI, Bedrock and Azure OpenAI all need a cloud account. If you "
        "only want Gemini models, an AI Studio key is free and simpler."
    )
    print()
    for key in ("vertexai", "bedrock", "azure"):
        print(f"   {PROVIDERS[key].label}: {PROVIDERS[key].free_tier}")

    if confirm("   Set up Google Cloud (Vertex AI) now?", False, assume_yes):
        _google(assume_yes)

    if confirm("   Set up AWS (Bedrock) now?", False, assume_yes):
        if command_exists("aws"):
            run_command(["aws", "configure"])
            run_command(["aws", "sts", "get-caller-identity"])
        else:
            print("   install the AWS CLI: https://aws.amazon.com/cli/")

    if confirm("   Set up Azure now?", False, assume_yes):
        if command_exists("az"):
            run_command(["az", "login"])
            run_command(["az", "account", "show"])
        else:
            print("   install the Azure CLI: https://learn.microsoft.com/cli/azure/")

    for key in ("bedrock", "azure"):
        provider = PROVIDERS[key]
        if provider.manual:
            print()
            print(colour(f"-- {provider.label}: manual step", BOLD))
            print(f"   {provider.manual}")


def _google(assume_yes: bool) -> None:
    if not command_exists("gcloud"):
        print("   install the gcloud CLI: https://cloud.google.com/sdk/docs/install")
        return

    run_command(["gcloud", "auth", "application-default", "login"])
    if assume_yes:
        # --non-interactive promises never to prompt, and there is no sensible
        # default for a project id.
        return
    try:
        project = input("   GCP project id: ").strip()
    except EOFError:
        return
    if not project:
        return

    run_command(["gcloud", "config", "set", "project", project])
    run_command(["gcloud", "auth", "application-default", "set-quota-project", project])
    # A project id is not a secret, but it is environment-specific; store it in
    # .env rather than anywhere tracked.
    write_env_var("GOOGLE_CLOUD_PROJECT", project)
    print(colour("   wrote GOOGLE_CLOUD_PROJECT to .env", GREEN))
