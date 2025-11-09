import re
import phonenumbers

class PrivacyAwareAnonymizer:
    """
    Anonymizer that removes personally identifiable information (PII)
    while preserving job-relevant content for BERT-based matching.

    Removes:
    - Names (proper nouns detection)
    - Email addresses
    - Phone numbers
    - URLs and social media
    - Physical addresses
    - Universities/institutions
    - Age, gender, marital status
    - ID numbers

    Preserves:
    - Skills and technologies
    - Job titles and roles
    - Experience descriptions
    - Projects and achievements
    - Education level (without institution names)
    - City/region (geographic area)
    """
    PLACEHOLDER_EMAIL = '[EMAIL]'
    PLACEHOLDER_PHONE = '[PHONE]'
    PLACEHOLDER_URL = '[URL]'
    PLACEHOLDER_SOCIAL = '[SOCIAL]'
    PLACEHOLDER_ID = '[ID]'
    PLACEHOLDER_NAME = '[NAME]'
    PLACEHOLDER_EDUCATION = '[EDUCATION]'

    def __init__(self):
        """Initialize with patterns for PII detection."""
        self._compile_patterns()
        self._load_common_names()
        self._load_university_keywords()

    def anonymize(self, text: str) -> str:
        """
        Remove PII while preserving job-relevant information.

        Args:
            text: Raw CV/resume text

        Returns:
            Anonymized text safe for processing
        """
        if not text or not isinstance(text, str):
            return ""

        anonymized = text

        # Step 1: Remove explicit PII
        anonymized = self._remove_emails(anonymized)
        anonymized = self._remove_phone_numbers(anonymized)
        anonymized = self._remove_urls(anonymized)
        anonymized = self._remove_social_media(anonymized)

        # Step 2: Remove personal identifiers
        anonymized = self._remove_id_numbers(anonymized)
        anonymized = self._remove_addresses(anonymized)

        # Step 3: Remove demographic information
        anonymized = self._remove_age_references(anonymized)
        anonymized = self._remove_gender_references(anonymized)
        anonymized = self._remove_marital_status(anonymized)

        # Step 4: Anonymize education (keep level, remove institution)
        anonymized = self._anonymize_education(anonymized)

        # Step 5: Remove names (careful to preserve technical terms)
        anonymized = self._remove_personal_names(anonymized)

        # Step 6: Clean up formatting
        anonymized = self._cleanup_formatting(anonymized)

        return anonymized.strip()

    def _compile_patterns(self):
        """Compile regex patterns for efficient matching."""
        self.email_pattern = re.compile(
            r'\b[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}\b',
            re.IGNORECASE
        )

        self.url_pattern = re.compile(
            r'https?://\S+|www\.\S+',
            re.IGNORECASE
        )

        self.social_patterns = [
            re.compile(r'(?<=\s)@\w+(?=\s|$)', re.IGNORECASE),
            re.compile(r'linkedin\.com/in/[\w-]+', re.IGNORECASE),
            re.compile(r'github\.com/[\w-]+', re.IGNORECASE),
            re.compile(r'twitter\.com/[\w-]+', re.IGNORECASE),
        ]

        self.id_patterns = [
            re.compile(r'\b\d{9,}\b'),
            re.compile(r'\b[A-Z]{2,3}\d{6,}\b'),
        ]

        self.age_patterns = [
            re.compile(r'\b\d{1,2}\s*(?:years?\s+old|a[ñn]os?)\b', re.IGNORECASE),
            re.compile(r'\b(?:age|edad):\s*\d{1,2}\b', re.IGNORECASE),
            re.compile(r'\bborn\s+(?:in|on)\s+\d{4}\b', re.IGNORECASE),
            re.compile(r'\bnacid[oa]\s+(?:en|el)\s+\d{4}\b', re.IGNORECASE),
        ]

        self.gender_patterns = [
            re.compile(r'\b(?:male|female|hombre|mujer|masculino|femenino|género|gender):\s*\w+\b', re.IGNORECASE),
            re.compile(r'\b(?:sex|sexo):\s*\w+\b', re.IGNORECASE),
        ]

        self.marital_patterns = [
            re.compile(r'\b(?:single|married|divorced|casad[oa]|solter[oa]|divorciad[oa]|viud[oa])\b', re.IGNORECASE),
            re.compile(r'\b(?:marital status|estado civil):\s*\w+\b', re.IGNORECASE),
        ]

    def _load_common_names(self):
        """Load common first/last names for detection."""
        self.common_first_names = {
            'juan', 'maría', 'jose', 'carlos', 'ana', 'luis', 'pedro', 'carmen',
            'john', 'mary', 'michael', 'sarah', 'david', 'jennifer', 'robert',
            'james', 'william', 'richard', 'thomas', 'daniel', 'jessica', 'emily'
        }

    def _load_university_keywords(self):
        """Load keywords indicating educational institutions."""
        self.university_keywords = {
            'university', 'universidad', 'college', 'instituto', 'institute',
            'school', 'escuela', 'academy', 'academia', 'polytechnic', 'politécnico'
        }

    def _remove_emails(self, text: str) -> str:
        """Remove email addresses."""
        return self.email_pattern.sub(PrivacyAwareAnonymizer.PLACEHOLDER_EMAIL, text)

    @staticmethod
    def _remove_phone_numbers(text: str) -> str:
        """Remove phone numbers in various formats."""
        result = text

        normalized = re.sub(r'[()\-.]', ' ', result)

        found_numbers = []
        for match in phonenumbers.PhoneNumberMatcher(normalized, "US"):
            found_numbers.append(match.raw_string.strip())

        for match in phonenumbers.PhoneNumberMatcher(normalized, "CO"):
            found_numbers.append(match.raw_string.strip())

        for number in found_numbers:
            if number:
                escaped = re.escape(number)
                result = re.sub(escaped, PrivacyAwareAnonymizer.PLACEHOLDER_PHONE, result)

        phone_regex = re.compile(
            r'(?:\+?\d{1,3}[\s\-.])?(?:\(?\d{2,4}\)?[\s\-.]?){2,4}\d{2,4}(?=\s|$)',
            re.IGNORECASE
        )

        safe_result = []
        for line in result.split('\n'):
            if not re.search(r'\b(?:python|aws|ec2|micro|version)\b', line, re.IGNORECASE):
                line = phone_regex.sub(PrivacyAwareAnonymizer.PLACEHOLDER_PHONE, line)
            safe_result.append(line)

        return '\n'.join(safe_result)

    def _remove_urls(self, text: str) -> str:
        """Remove URLs."""
        return self.url_pattern.sub(PrivacyAwareAnonymizer.PLACEHOLDER_URL, text)

    def _remove_social_media(self, text: str) -> str:
        """Remove social media handles and profiles."""
        result = text
        for pattern in self.social_patterns:
            result = pattern.sub(PrivacyAwareAnonymizer.PLACEHOLDER_SOCIAL, result)
        return result

    def _remove_id_numbers(self, text: str) -> str:
        """Remove identification numbers."""
        result = text
        for pattern in self.id_patterns:
            result = pattern.sub(PrivacyAwareAnonymizer.PLACEHOLDER_ID, result)
        return result

    @staticmethod
    def _remove_addresses(text: str) -> str:
        """
        Remove physical addresses while keeping city/region.
        This is complex - we use heuristics to detect address patterns.
        """
        address_pattern = re.compile(
            r'\b\d+\s+[A-Z][a-z]+\s+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Calle|Avenida|Carrera)\b',
            re.IGNORECASE
        )
        result = address_pattern.sub('[ADDRESS]', text)

        zip_pattern = re.compile(r'\b\d{5}(?:-\d{4})?\b')
        result = zip_pattern.sub('', result)

        return result

    def _remove_age_references(self, text: str) -> str:
        """Remove age-related information."""
        result = text
        for pattern in self.age_patterns:
            result = pattern.sub('[AGE]', result)
        return result

    def _remove_gender_references(self, text: str) -> str:
        """Remove gender information."""
        result = text
        for pattern in self.gender_patterns:
            result = pattern.sub('', result)
        return result

    def _remove_marital_status(self, text: str) -> str:
        """Remove marital status information."""
        result = text
        for pattern in self.marital_patterns:
            result = pattern.sub('', result)
        return result

    def _anonymize_education(self, text: str) -> str:
        """
        Replace university names with generic labels.
        Keep degree level but remove institution name.
        """
        lines = text.split('\n')
        anonymized_lines = []

        for line in lines:
            line_lower = line.lower()
            contains_university = any(keyword in line_lower for keyword in self.university_keywords)

            if contains_university:
                degree_keywords = [
                    'bachelor', 'master', 'phd', 'doctorate', 'licenciatura',
                    'maestría', 'doctorado', 'ingeniería', 'engineering'
                ]

                preserved_parts = []
                for keyword in degree_keywords:
                    if keyword in line_lower:
                        preserved_parts.append(keyword.title())

                if preserved_parts:
                    anonymized_lines.append(' '.join(preserved_parts))
                else:
                    anonymized_lines.append(PrivacyAwareAnonymizer.PLACEHOLDER_EDUCATION)
            else:
                anonymized_lines.append(line)

        return '\n'.join(anonymized_lines)

    @staticmethod
    def _remove_personal_names(text: str) -> str:
        """
        Remove personal names while preserving technical terms.
        This is the most challenging part - we use heuristics.
        """
        lines = text.split('\n')
        anonymized_lines = []

        for i, line in enumerate(lines):
            if PrivacyAwareAnonymizer.PLACEHOLDER_EMAIL in line or PrivacyAwareAnonymizer.PLACEHOLDER_PHONE in line or PrivacyAwareAnonymizer.PLACEHOLDER_SOCIAL in line or PrivacyAwareAnonymizer.PLACEHOLDER_URL in line:
                anonymized_lines.append(line)
                continue
            if i < 3:
                words = line.split()
                if (
                    i < 3 and len(words) <= 4
                    and not any(w.lower() in {"summary", "profile", "skills"} for w in words)
                ):
                    anonymized_lines.append(PrivacyAwareAnonymizer.PLACEHOLDER_NAME)
                    continue

            line = re.sub(r'\b(?:name|nombre):\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', PrivacyAwareAnonymizer.PLACEHOLDER_NAME, line, flags=re.IGNORECASE)

            anonymized_lines.append(line)

        return '\n'.join(anonymized_lines)

    @staticmethod
    def _cleanup_formatting(text: str) -> str:
        """Clean up formatting after anonymization."""
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'\n\s*\[\w+]\s*\n', '\n', text)
        text = re.sub(r'\n\s+\n', '\n\n', text)

        return text
