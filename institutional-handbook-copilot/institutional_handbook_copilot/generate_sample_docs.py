"""
Generates clean sample institutional PDF documents for testing version conflict detection.
"""

import os


def create_minimal_pdf(filepath: str, pages_text: list):
    """
    Creates a standard, valid PDF 1.4 document containing the provided text on each page.
    Requires no external dependencies.
    """
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    
    objects = []
    
    # Obj 1: Catalog
    # Obj 2: Pages
    # Then for each page: Page obj, Content stream obj, Font obj
    
    font_obj_idx = 3
    page_obj_indices = []
    content_obj_indices = []
    
    current_idx = 4
    for _ in pages_text:
        page_obj_indices.append(current_idx)
        content_obj_indices.append(current_idx + 1)
        current_idx += 2
        
    catalog_obj = f"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
    
    kids_str = " ".join(f"{idx} 0 R" for idx in page_obj_indices)
    pages_obj = f"2 0 obj\n<< /Type /Pages /Kids [{kids_str}] /Count {len(pages_text)} >>\nendobj\n"
    
    font_obj = f"3 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
    
    body = [catalog_obj, pages_obj, font_obj]
    
    for i, text in enumerate(pages_text):
        p_idx = page_obj_indices[i]
        c_idx = content_obj_indices[i]
        
        # Build text stream with positioning
        lines = text.split('\n')
        stream_cmds = ["BT", "/F1 12 Tf", "50 750 Td", "16 TL"]
        for line in lines:
            safe_line = line.replace('\\', '\\\\').replace('(', '\\(').replace(')', '\\)')
            stream_cmds.append(f"({safe_line}) '")
        stream_cmds.append("ET")
        stream_content = "\n".join(stream_cmds)
        
        page_obj = (
            f"{p_idx} 0 obj\n"
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]\n"
            f"   /Contents {c_idx} 0 R\n"
            f"   /Resources << /Font << /F1 3 0 R >> >>\n"
            f">>\nendobj\n"
        )
        
        stream_len = len(stream_content.encode('latin1'))
        content_obj = (
            f"{c_idx} 0 obj\n"
            f"<< /Length {stream_len} >>\n"
            f"stream\n{stream_content}\nendstream\nendobj\n"
        )
        
        body.append(page_obj)
        body.append(content_obj)
        
    # Assemble PDF
    pdf_bytes = b"%PDF-1.4\n"
    offsets = [0]
    
    for obj_str in body:
        offsets.append(len(pdf_bytes))
        pdf_bytes += obj_str.encode('latin1')
        
    xref_offset = len(pdf_bytes)
    xref_str = f"xref\n0 {len(body) + 1}\n0000000000 65535 f \n"
    for off in offsets[1:]:
        xref_str += f"{off:010d} 00000 n \n"
        
    trailer_str = (
        f"trailer\n<< /Size {len(body) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n"
    )
    pdf_bytes += (xref_str + trailer_str).encode('latin1')
    
    with open(filepath, "wb") as f:
        f.write(pdf_bytes)
    print(f"Generated PDF: {filepath} ({len(pages_text)} pages)")


def generate_sample_handbooks(dest_dir: str = "sample_docs"):
    os.makedirs(dest_dir, exist_ok=True)
    
    # 2025 Handbook
    p2025_page1 = (
        "UNIVERSITY ACADEMIC REGULATIONS & POLICIES\n"
        "Official Handbook 2025\n"
        "Academic Year: 2024-2025\n"
        "Effective Date: August 1, 2024\n"
        "\n"
        "SECTION 3: ATTENDANCE RULES\n"
        "Minimum attendance requirement: 75%.\n"
        "Students who fail to maintain 75% attendance in any course will be debarred from the end-semester examinations.\n"
        "\n"
        "SECTION 4: PROMOTION AND BACKLOG POLICY\n"
        "Students may have up to 4 backlogs for promotion to the next academic year.\n"
        "A student having more than 4 backlogs shall repeat the entire academic semester.\n"
        "\n"
        "SECTION 5: GRADING SYSTEM\n"
        "The minimum passing grade for any course is Grade D (40 marks)."
    )
    
    p2025_page2 = (
        "UNIVERSITY POLICIES 2025 - CONTINUED\n"
        "SECTION 6: LEAVE AND ABSENCE\n"
        "Medical leave must be submitted within 7 days of absence.\n"
        "Maximum casual leave allowed per semester is 10 days."
    )
    
    path_2025 = os.path.join(dest_dir, "handbook_2025.pdf")
    create_minimal_pdf(path_2025, [p2025_page1, p2025_page2])
    
    # 2026 Handbook (Newer Version with changes!)
    p2026_page1 = (
        "UNIVERSITY ACADEMIC REGULATIONS & POLICIES\n"
        "Official Handbook 2026\n"
        "Academic Year: 2025-2026\n"
        "Effective Date: August 1, 2025\n"
        "\n"
        "SECTION 3: ATTENDANCE RULES\n"
        "Minimum attendance requirement: 80%.\n"
        "Due to updated university accreditation standards, students must now maintain at least 80% attendance in all courses.\n"
        "\n"
        "SECTION 4: PROMOTION AND BACKLOG POLICY\n"
        "Students may have up to 2 backlogs for promotion to the next academic year.\n"
        "The previous threshold of 4 backlogs has been superseded to improve graduation rates.\n"
        "\n"
        "SECTION 5: GRADING SYSTEM\n"
        "The minimum passing grade for any course is Grade C (50 marks)."
    )
    
    p2026_page2 = (
        "UNIVERSITY POLICIES 2026 - CONTINUED\n"
        "SECTION 6: LEAVE AND ABSENCE\n"
        "Medical leave must be submitted within 5 days of absence.\n"
        "Maximum casual leave allowed per semester is 7 days."
    )
    
    path_2026 = os.path.join(dest_dir, "handbook_2026.pdf")
    create_minimal_pdf(path_2026, [p2026_page1, p2026_page2])
    
    return path_2025, path_2026


if __name__ == "__main__":
    generate_sample_handbooks("sample_docs")
