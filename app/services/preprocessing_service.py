class CVPreprocessingService:
    """
    Convierte un CV en formato JSON estructurado en texto plano
    listo para análisis semántico (embeddings).
    """

    def cv_to_text(self, cv_json: dict) -> str:
        text_parts = []

        # Resumen general
        text_parts.append(cv_json.get("personal_info", {}).get("summary", ""))

        # Experiencia laboral
        for exp in cv_json.get("experience", []):
            text_parts.append(exp.get("title", ""))
            text_parts.extend(exp.get("responsibilities", []))
            text_parts.extend(exp.get("technical_environment", {}).get("technologies", []))

        # Educación
        for edu in cv_json.get("education", []):
            text_parts.append(edu.get("degree", {}).get("field", ""))

        # Habilidades técnicas
        for group in cv_json.get("skills", {}).get("technical", {}).values():
            for skill in group:
                text_parts.append(skill.get("name", ""))

        # Proyectos
        for proj in cv_json.get("projects", []):
            text_parts.append(proj.get("description", ""))

        # Limpieza final
        text = " ".join([p for p in text_parts if p and p not in ("Unknown", "Not Provided")])
        return text.strip()
