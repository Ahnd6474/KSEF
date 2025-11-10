import os
import py3Dmol

def show_xyz(
    path,
    width=500,
    height=500,
    style='stick',
    style_opts=None,
    legend=None,
    out_html=None,
):
    """
    .xyz 파일을 py3Dmol로 시각화 + 색상 legend까지 같이 보여주는 함수.

    - 노트북(Jupyter/Colab)에서는 바로 렌더링.
    - 스크립트에서는 HTML 파일로 저장 후 경로를 출력/반환.

    legend: [(label, color), ...] 리스트
        예: [("Carbon (C)", "gray"), ("Oxygen (O)", "red")]
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"No such file: {path}")

    # 기본 legend 없으면 대충 원소 기준 예시 하나 깔아줌
    if legend is None:
        legend = [
            ("C (carbon)", "gray"),
            ("O (oxygen)", "red"),
            ("N (nitrogen)", "blue"),
        ]

    with open(path, 'r') as f:
        xyz_data = f.read()

    if style_opts is None:
        style_opts = {}

    view = py3Dmol.view(width=width, height=height)
    view.addModel(xyz_data, 'xyz')
    view.setStyle({style: style_opts})
    view.zoomTo()

    # py3Dmol이 만들어주는 기본 HTML
    base_html = view._make_html()

    # legend용 HTML 조각
    legend_items = []
    for label, color in legend:
        legend_items.append(
            f"""
            <div style="display:flex;align-items:center;gap:6px;margin:2px 0;">
              <span style="display:inline-block;width:12px;height:12px;
                           background:{color};border:1px solid #999;"></span>
              <span>{label}</span>
            </div>
            """
        )
    legend_html = """
    <div style="margin-top:8px;font-family:Arial, sans-serif;font-size:13px;">
      <b>Legend</b>
      {items}
    </div>
    """.format(items="\n".join(legend_items))

    # base_html 안에 legend를 body 끝나기 전에 끼워 넣기
    if '</body>' in base_html:
        full_html = base_html.replace('</body>', legend_html + '</body>')
    else:
        full_html = base_html + legend_html

    # 노트북이면 바로 렌더링, 아니면 파일로 저장
    try:
        # Jupyter 여부 판단
        get_ipython  # type: ignore
        from IPython.display import HTML, display  # type: ignore

        display(HTML(full_html))
        return None
    except NameError:
        if out_html is None:
            out_html = os.path.splitext(path)[0] + '_view.html'
        with open(out_html, 'w', encoding='utf-8') as f:
            f.write(full_html)
        print(f"✅ Visualization with legend saved to: {out_html}")
        return out_html


show_xyz('qm9/temp/qm9/dsgdb9nsd_000003.xyz',style='sphere')
