"""Render the write-up's canonical SVGs to publication PNGs; no game imports.

Run with /usr/bin/python3 -B src/farm_neat/tooling/export_figure.py.
Requires system Cairo and librsvg 2.0; installs nothing.
SVG is the editable source. PNG is a deterministic derived view for this renderer.
"""
import ctypes as c
from ctypes.util import find_library
from pathlib import Path

FIGURES = ("architecture", "interaction", "training-objectives", "decision-step", "exemplar-board", "training-loop")


class Rectangle(c.Structure):
    _fields_ = [(name, c.c_double) for name in ("x", "y", "width", "height")]


def library(name):
    path = find_library(name)
    if not path:
        raise SystemExit(f"Missing system library: {name}; SVG remains usable directly")
    return c.CDLL(path)


def bind(lib, name, result, *arguments):
    function = getattr(lib, name)
    function.restype, function.argtypes = result, arguments
    return function


def export(name):
    svg, cairo, objects = library("rsvg-2"), library("cairo"), library("gobject-2.0")
    pointer, number = c.c_void_p, c.c_double
    load = bind(svg, "rsvg_handle_new_from_file", pointer, c.c_char_p, pointer)
    dimensions = bind(svg, "rsvg_handle_get_intrinsic_size_in_pixels", c.c_int,
                      pointer, c.POINTER(number), c.POINTER(number))
    render = bind(svg, "rsvg_handle_render_document", c.c_int,
                  pointer, pointer, c.POINTER(Rectangle), pointer)
    surface_create = bind(cairo, "cairo_image_surface_create", pointer,
                          c.c_int, c.c_int, c.c_int)
    context_create = bind(cairo, "cairo_create", pointer, pointer)
    scale = bind(cairo, "cairo_scale", None, pointer, number, number)
    write = bind(cairo, "cairo_surface_write_to_png", c.c_int, pointer, c.c_char_p)
    context_destroy = bind(cairo, "cairo_destroy", None, pointer)
    surface_destroy = bind(cairo, "cairo_surface_destroy", None, pointer)
    unref = bind(objects, "g_object_unref", None, pointer)
    root = Path(__file__).resolve().parents[3]
    source = root / "kagglewriteup" / f"{name}.svg"
    target = source.with_suffix(".png")
    handle = load(str(source).encode(), None)
    if not handle:
        raise SystemExit(f"Could not load {source.name}")
    surface = context = None
    try:
        width, height = number(), number()
        valid = dimensions(handle, c.byref(width), c.byref(height))
        if not valid or not (0 < width.value <= 3000 and 0 < height.value <= 3000):
            raise SystemExit("Expected bounded figure with explicit intrinsic dimensions")
        surface = surface_create(0, int(width.value * 1.5), int(height.value * 1.5))
        context = context_create(surface)
        scale(context, 1.5, 1.5)
        viewport = Rectangle(0, 0, width.value, height.value)
        if not render(handle, context, c.byref(viewport), None):
            raise SystemExit("SVG rendering failed")
        if write(surface, str(target).encode()) != 0:
            raise SystemExit("Could not write architecture.png")
    finally:
        if context:
            context_destroy(context)
        if surface:
            surface_destroy(surface)
        unref(handle)
    print(f"Rendered {target.relative_to(root)} from SVG ({target.stat().st_size} bytes)")


def main():
    for name in FIGURES:
        export(name)


if __name__ == "__main__":
    main()
