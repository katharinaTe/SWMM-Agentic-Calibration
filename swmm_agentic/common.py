"""Shared setup: imports, SWMM executable, LLM and printing helpers.

The code in this module is taken unchanged from the notebooks
(cells 1-10, identical in all six notebooks) (v1.0.0, https://doi.org/10.5281/zenodo.23478514).
Only the cell boundaries were removed.
"""

# Import packages and configure SWMM path

import os
import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spotpy
from spotpy.analyser import get_simulation_fields, get_minlikeindex

from swmm_api import SwmmInput
from swmm_api.input_file.macros import delete_subcatchment
from swmm_api.output_file import VARIABLES
from swmm_api.run_swmm.run_temporary import swmm5_run_temporary

# LangChain / OpenAI-Integration
#from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
#from langchain.agents import AgentExecutor, create_openai_tools_agent

from langchain_core.messages import convert_to_messages

from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor

import time
import resource

from swmm_api import CONFIG

CONFIG["exe_path"] = "/home/katharina_teuber/Stormwater-Management-Model/build/install/bin/runswmm"
# oder:
# CONFIG.exe_path = "/home/..."

print("exe_path =", CONFIG["exe_path"])
# oder:
print("exe_path_attr =", CONFIG.exe_path)

# Choose LLM-model

llm = ChatOllama(
model="gpt-oss:20b",
temperature=0.1,
base_url="http://localhost:11434")

def pretty_print_message(message, indent=False):
    pretty_message = message.pretty_repr(html=True)
    if not indent:
        print(pretty_message)
        return

    indented = "\n".join("\t" + c for c in pretty_message.split("\n"))
    print(indented)


def pretty_print_messages(update, last_message=False):
    is_subgraph = False
    if isinstance(update, tuple):
        ns, update = update
        # skip parent graph updates in the printouts
        if len(ns) == 0:
            return

        graph_id = ns[-1].split(":")[0]
        print(f"Update from subgraph {graph_id}:")
        print("\n")
        is_subgraph = True

    for node_name, node_update in update.items():
        update_label = f"Update from node {node_name}:"
        if is_subgraph:
            update_label = "\t" + update_label

        print(update_label)
        print("\n")

        # added code - start -
        if not isinstance(node_update, dict):
            # optional: show raw data or skip
            # print("\t(non-dict update, skipping)\n" if is_subgraph else "(non-dict update, skipping)\n")
            continue

        msgs_raw = node_update.get("messages")
        if not msgs_raw:
            # optional: info which data has been skipped
            # print("\t(no messages in update, skipping)\n" if is_subgraph else "(no messages in update, skipping)\n")
            continue
        
        # added code - end -
        messages = convert_to_messages(node_update["messages"])
        if last_message:
            pass
            #messages = messages[-1:]

        for m in messages:
            pretty_print_message(m, indent=is_subgraph)
        print("\n")
