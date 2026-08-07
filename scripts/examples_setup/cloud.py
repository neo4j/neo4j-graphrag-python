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
"""Tier 3: Vertex AI, Bedrock and Azure OpenAI.

Branches on who is running it. The Neo4j-internal path reads an Aura dev
environment via omni; the other is the plain vendor flow. Neither is right for
both audiences, so the installer asks rather than guessing.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from .console import (
    BOLD,
    DIM,
    GREEN,
    RED,
    YELLOW,
    colour,
    confirm,
    heading,
    run_command,
)
from .envfile import write_env_var
from .probes import command_exists
from .providers import PROVIDERS, TIER_NAMES


def tier_3_cloud(assume_yes: bool, cloud_profile: Optional[str]) -> None:
    heading(f"Tier 3: {TIER_NAMES[3]}")
    print(
        "Vertex AI, Bedrock and Azure OpenAI all need a cloud account. If you "
        "only want Gemini models, tier 1 (AI Studio) is free and simpler."
    )

    profile = cloud_profile or _choose_cloud_profile(assume_yes)
    if profile == "aura":
        _tier_3_aura(assume_yes)
    else:
        _tier_3_own(assume_yes)

    for key in ("bedrock", "azure"):
        provider = PROVIDERS[key]
        if provider.manual:
            print()
            print(colour(f"-- {provider.label}: manual step", BOLD))
            print(f"   {provider.manual}")


def _choose_cloud_profile(assume_yes: bool) -> str:
    """Neo4j-internal Aura dev environment, or the user's own cloud accounts."""
    has_omni = command_exists("omni") or (Path.home() / "go/bin/omni").exists()
    if assume_yes:
        return "aura" if has_omni else "own"
    print()
    if has_omni:
        print("   'omni' is on this machine, so an Aura dev environment is likely.")
    print("   1) Neo4j internal - an Aura dev environment, via omni")
    print("   2) Your own GCP / AWS / Azure accounts")
    default = "1" if has_omni else "2"
    try:
        answer = input(f"   Which applies? [{default}] ").strip() or default
    except EOFError:
        answer = default
    return "aura" if answer.startswith("1") else "own"


def _tier_3_aura(assume_yes: bool) -> None:
    omni = shutil.which("omni") or str(Path.home() / "go/bin/omni")
    if not Path(omni).exists():
        print(colour("   omni not found; falling back to your own accounts", YELLOW))
        _tier_3_own(assume_yes)
        return

    try:
        env_name = input("   Aura dev environment name: ").strip()
    except EOFError:
        return
    if not env_name:
        return

    print(colour(f"  $ {omni} environments info {env_name} --json", DIM))
    result = subprocess.run(
        [omni, "environments", "info", env_name, "--json"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(
            colour(f"   could not read that environment: {result.stderr.strip()}", RED)
        )
        return

    try:
        spec = json.loads(result.stdout).get("spec", {})
    except json.JSONDecodeError:
        print(colour("   unexpected output from omni", RED))
        return

    # Read live and keep in memory. These identifiers change whenever the
    # environment is rebuilt, so they are never written to .env or to any file.
    gcp_project = spec.get("gcp_project_id")
    if gcp_project:
        print(f"   GCP project: {gcp_project}")
        if confirm("   Configure gcloud ADC for it?", True, assume_yes):
            run_command([omni, "auth", "login", "gcp"])
            if command_exists("gcloud"):
                run_command(
                    [
                        "gcloud",
                        "auth",
                        "application-default",
                        "set-quota-project",
                        str(gcp_project),
                    ]
                )
                print(
                    "   export GOOGLE_CLOUD_PROJECT="
                    f"{gcp_project} in your shell for this session"
                )
    print(
        colour(
            "   Vertex AI is a managed API - it works while the environment is "
            "SCALED_DOWN, so there is no need to run 'omni environments up'.",
            DIM,
        )
    )

    if confirm("   Log in to AWS for Bedrock?", False, assume_yes):
        run_command([omni, "auth", "login", "aws"])
        if command_exists("aws"):
            run_command(["aws", "sts", "get-caller-identity"])


def _tier_3_own(assume_yes: bool) -> None:
    print()
    print(colour("-- Your own cloud accounts", BOLD))
    for key in ("vertexai", "bedrock", "azure"):
        print(f"   {PROVIDERS[key].label}: {PROVIDERS[key].free_tier}")

    if confirm("   Set up Google Cloud (Vertex AI) now?", False, assume_yes):
        if command_exists("gcloud"):
            run_command(["gcloud", "auth", "application-default", "login"])
            try:
                project = input("   GCP project id: ").strip()
            except EOFError:
                project = ""
            if project:
                run_command(["gcloud", "config", "set", "project", project])
                run_command(
                    [
                        "gcloud",
                        "auth",
                        "application-default",
                        "set-quota-project",
                        project,
                    ]
                )
                # A project id is not a secret, but it is environment-specific;
                # store it in .env rather than anywhere tracked.
                write_env_var("GOOGLE_CLOUD_PROJECT", project)
                print(colour("   wrote GOOGLE_CLOUD_PROJECT to .env", GREEN))
        else:
            print(
                "   install the gcloud CLI: https://cloud.google.com/sdk/docs/install"
            )

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
