"""The MCP server: tool list and schemas, results equal to the API, errors, and a real stdio session."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import pytest

pytest.importorskip("mcp")

from liftingline import api, mcp_server  # noqa: E402
from liftingline.io.presets import CESSNA_172S  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MPH = 0.44704
TOOLS = {"list_examples", "describe_project", "analyze_wing", "sweep", "max_lift", "trim", "run_scenario",
         "compare_real_aircraft", "export_stl", "self_test", "get_documentation"}


@pytest.fixture(scope="module")
def server():
    return mcp_server.build_server()


def call(server, name, args):
    out = asyncio.run(server.call_tool(name, args))
    content, structured = out if isinstance(out, tuple) else (out, None)
    if structured is None:
        return json.loads(content[0].text)
    return structured.get("result", structured) if set(structured) == {"result"} else structured


def test_tools_and_their_schemas(server):
    tools = {t.name: t for t in asyncio.run(server.list_tools())}
    assert set(tools) == TOOLS
    for t in tools.values():
        assert t.description and len(t.description) > 40, t.name
    props = tools["analyze_wing"].inputSchema["properties"]
    assert {"example", "project_file", "project", "alpha_deg", "speed", "speed_unit", "altitude_m", "engine",
            "flaps", "slats", "output"} <= set(props)
    assert all(props[k].get("description") for k in ("example", "alpha_deg", "speed_unit"))
    assert set(tools["run_scenario"].inputSchema["properties"]) >= {"v_start", "v_end", "v_rotate", "ramp", "mass_kg",
                                                                    "time_model", "speed_unit", "max_frames"}


def test_analyze_equals_the_api(server):
    got = call(server, "analyze_wing", {"example": "Cessna 172S", "engine": "builtin", "altitude_m": 0, "alpha_deg": 3})
    st = api.apply_overrides(api.load_project(example=CESSNA_172S), engine="builtin", altitude_m=0.0, alpha_deg=3.0)
    want = api.analyze(st, spanwise=False).data
    assert got["schema"] == "liftingline.analysis/1" and got["totals"] == want["totals"]
    text = call(server, "analyze_wing", {"example": "rectangular", "engine": "builtin", "output": "text"})
    assert text["format"] == "text" and "Lift L" in text["content"]


def test_scenario_tool_in_user_units(server):
    d = call(server, "run_scenario", {"example": "Cessna 172S", "engine": "builtin", "altitude_m": 0,
                                      "speed_unit": "mph", "v_start": 0, "v_end": 90, "v_rotate": 60,
                                      "ramp": "target", "ramp_target_deg": 10, "time_model": True, "points": 21,
                                      "max_frames": 11})
    assert 60 * MPH < d["liftoff"]["airspeed_mps"] < 90 * MPH and d["liftoff"]["t_s"] > 0
    assert len(d["frames"]) <= 11 and d["inputs"]["speed_unit"] == "mph"


def test_errors_are_readable(server):
    from mcp.server.fastmcp.exceptions import ToolError

    with pytest.raises(ToolError, match="no example called"):
        call(server, "analyze_wing", {"example": "concorde"})
    with pytest.raises(ToolError, match="rotation speed"):
        call(server, "run_scenario", {"example": "rectangular", "engine": "builtin", "v_end": 50, "v_rotate": 60})
    with pytest.raises(ToolError, match="no CSV"):
        call(server, "trim", {"example": "rectangular", "engine": "builtin", "target_CL": 0.4, "output": "csv"})


def test_documentation_tool_and_resource(server):
    doc = call(server, "get_documentation", {})
    doc = doc if isinstance(doc, str) else doc["result"]
    assert "liftingline.analysis/1" in doc and "liftingline.scenario/1" in doc
    contents = asyncio.run(server.read_resource(mcp_server.DOC_URI))
    assert "liftingline.analysis/1" in list(contents)[0].content


def test_a_real_stdio_session(tmp_path):
    """Start ``python -m liftingline mcp`` as a subprocess and talk to it like an MCP client does."""
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    env = dict(os.environ, PYTHONPATH=str(ROOT))
    params = StdioServerParameters(command=sys.executable, args=["-m", "liftingline", "mcp"], env=env, cwd=str(tmp_path))

    async def session():
        async with stdio_client(params) as (read, write), ClientSession(read, write) as s:
            init = await s.initialize()
            tools = await s.list_tools()
            res = await s.call_tool("analyze_wing", {"example": "Rectangular wing, AR 8", "engine": "builtin"})
            bad = await s.call_tool("analyze_wing", {"example": "concorde"})
            stl = await s.call_tool("export_stl", {"example": "rectangular", "path": str(tmp_path / "w.stl"),
                                                   "quality": "coarse"})
            return init, tools, res, bad, stl

    init, tools, res, bad, stl = asyncio.run(asyncio.wait_for(session(), 180))
    assert init.serverInfo.name == "liftingline" and "lifting-line" in (init.instructions or "").lower()
    assert {t.name for t in tools.tools} == TOOLS
    assert not res.isError
    data = res.structuredContent or json.loads(res.content[0].text)
    assert data["totals"]["CL"] > 0.2 and data["section_data"].startswith("Built-in")
    assert bad.isError and "no example called" in bad.content[0].text
    assert not stl.isError and (tmp_path / "w.stl").stat().st_size > 1000
