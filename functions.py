
"""
Wikipedia Math Fetcher for Jupyter Notebook
============================================
A utility for fetching Wikipedia content and rendering it beautifully in Jupyter,
with proper MathJax support for mathematical expressions.

Features:
- Converts Wikipedia LaTeX fragments to MathJax-compatible syntax
- Collapses stacked/vertical math tokens into inline expressions
- Graceful fallback to plain text in terminal environments
- Uses only standard library + wikipedia + IPython (no fragile dependencies)

Usage:
    from functions import wiki_math
    
    # In Jupyter Notebook:
    wiki_math("Singular value decomposition")
    
    # Or get the formatted content as a string:
    content = wiki_fetch("Singular value decomposition")
    print(content)
"""

import re
import wikipedia


# =============================================================================
# Environment Detection
# =============================================================================

def is_jupyter() -> bool:
    """
    Detect if code is running in a Jupyter Notebook environment.
    
    Returns:
        bool: True if running in Jupyter, False otherwise (terminal/script).
    """
    try:
        from IPython import get_ipython
        shell = get_ipython()
        if shell is None:
            return False
        # Check for Jupyter-specific shell types
        shell_name = shell.__class__.__name__
        return shell_name in ('ZMQInteractiveShell', 'Shell')
    except (ImportError, NameError):
        return False


# =============================================================================
# Text Cleaning & Math Conversion
# =============================================================================

def collapse_stacked_math(text: str) -> str:
    """
    Collapse vertically stacked single characters/symbols into inline expressions.
    
    Wikipedia's plain text extraction sometimes splits math like:
        m
        ×
        n
    This function collapses such patterns into: m × n
    
    Args:
        text: Raw text that may contain stacked math tokens.
        
    Returns:
        Text with collapsed math expressions.
    """
    lines = text.split('\n')
    result = []
    i = 0
    
    while i < len(lines):
        line = lines[i].strip()
        
        # Check if this looks like a stacked math sequence
        # (short lines of 1-3 chars that are math-like)
        if len(line) <= 3 and re.match(r'^[a-zA-Z0-9×÷±∓≤≥≠≈∈∉⊂⊃∪∩∧∨¬→←↔∀∃∂∇∫∑∏√∞αβγδεζηθικλμνξπρστυφχψωΓΔΘΛΞΠΣΦΨΩ=+\-*/^().,]+$', line):
            # Collect consecutive short math-like lines
            math_tokens = [line]
            j = i + 1
            while j < len(lines):
                next_line = lines[j].strip()
                if len(next_line) <= 3 and next_line and re.match(r'^[a-zA-Z0-9×÷±∓≤≥≠≈∈∉⊂⊃∪∩∧∨¬→←↔∀∃∂∇∫∑∏√∞αβγδεζηθικλμνξπρστυφχψωΓΔΘΛΞΠΣΦΨΩ=+\-*/^().,]+$', next_line):
                    math_tokens.append(next_line)
                    j += 1
                else:
                    break
            
            # If we found a sequence of 3+ math tokens, collapse them
            if len(math_tokens) >= 3:
                collapsed = ' '.join(math_tokens)
                result.append(collapsed)
                i = j
                continue
        
        result.append(lines[i])
        i += 1
    
    return '\n'.join(result)


def convert_displaystyle_latex(text: str) -> str:
    """
    Convert Wikipedia's {\\displaystyle ...} LaTeX fragments to MathJax format.
    
    Wikipedia often includes raw LaTeX like:
        {\\displaystyle m\\times n}
        {\\displaystyle \\mathbf{M}=U\\Sigma V^{*}}
    
    This converts them to: $m \\times n$ and $\\mathbf{M}=U\\Sigma V^{*}$
    
    Args:
        text: Text containing displaystyle LaTeX fragments.
        
    Returns:
        Text with MathJax-compatible $...$ syntax.
    """
    # Pattern for {\\displaystyle ...} blocks (handles nested braces)
    def replace_displaystyle(match):
        content = match.group(1)
        # Clean up the content
        content = content.strip()
        # Ensure proper spacing around operators
        content = re.sub(r'\\times', r' \\times ', content)
        content = re.sub(r'\\cdot', r' \\cdot ', content)
        content = re.sub(r'\s+', ' ', content)  # Normalize whitespace
        return f'${content.strip()}$'
    
    # Match {\\displaystyle ...} with balanced braces
    pattern = r'\{\\displaystyle\s+([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
    text = re.sub(pattern, replace_displaystyle, text)
    
    return text


def convert_inline_latex_fragments(text: str) -> str:
    """
    Convert remaining LaTeX-like fragments to MathJax inline math.
    
    Catches patterns like:
        \\mathbf{M}, \\Sigma, \\alpha, m\\times n
    
    Args:
        text: Text with LaTeX fragments not wrapped in $.
        
    Returns:
        Text with LaTeX wrapped in $...$ for MathJax rendering.
    """
    # Find LaTeX commands not already wrapped in $
    # Pattern: backslash followed by command name (and possibly arguments)
    def wrap_latex(match):
        # Check if already inside $ delimiters
        content = match.group(0)
        return f'${content}$'
    
    # Match LaTeX commands that aren't already in $...$
    # This is a simplified pattern that catches common cases
    patterns = [
        r'(?<!\$)\\mathbf\{[^}]+\}(?!\$)',
        r'(?<!\$)\\mathit\{[^}]+\}(?!\$)',
        r'(?<!\$)\\mathrm\{[^}]+\}(?!\$)',
        r'(?<!\$)\\[A-Za-z]+(?:\{[^}]*\})?(?!\$)',
    ]
    
    for pattern in patterns:
        # Only wrap if not already between $ signs
        text = re.sub(pattern, wrap_latex, text)
    
    # Clean up any double-wrapped math: $$...$$ -> $...$
    text = re.sub(r'\$\$([^$]+)\$\$', r'$\1$', text)
    
    # Clean up adjacent math blocks: $a$ $b$ -> $a \; b$
    text = re.sub(r'\$\s*\$', ' ', text)
    
    return text


def clean_wikipedia_text(text: str, for_jupyter: bool = True) -> str:
    """
    Clean and format Wikipedia text for display.
    
    Args:
        text: Raw Wikipedia text content.
        for_jupyter: If True, format for Jupyter/MathJax. If False, plain text.
        
    Returns:
        Cleaned and formatted text.
    """
    # Step 1: Collapse stacked math tokens
    text = collapse_stacked_math(text)
    
    # Step 2: Remove the ⁠ (word joiner) Unicode character that Wikipedia uses
    text = text.replace('\u2060', '')
    
    # Step 3: Remove duplicate math expressions (Wikipedia often has both Unicode and LaTeX)
    # Pattern: "X {\\displaystyle X}" -> just keep the displaystyle version
    text = re.sub(r'(\S+)\s+\{\\displaystyle\s+\1\s*\}', r'{\\displaystyle \1}', text)
    # Pattern: "X Y {\\displaystyle X Y}" or similar
    text = re.sub(r'([A-Za-z0-9×÷±≤≥≠≈∈∉⊂⊃∪∩∞Σσαβγδλμπθφω\s\*\^]+)\s+\{\\displaystyle\s+[^}]*\}', 
                  lambda m: '{\\displaystyle ' + m.group(1).strip() + '}' if len(m.group(1).strip()) <= 20 else m.group(0), 
                  text)
    
    if for_jupyter:
        # Step 4: Convert displaystyle LaTeX
        text = convert_displaystyle_latex(text)
        
        # Step 5: Convert remaining LaTeX fragments
        text = convert_inline_latex_fragments(text)
        
        # Step 6: Clean up common Unicode math symbols for consistency
        unicode_to_latex = {
            '×': r' \times ',
            '÷': r' \div ',
            '±': r' \pm ',
            '≤': r' \leq ',
            '≥': r' \geq ',
            '≠': r' \neq ',
            '≈': r' \approx ',
            '∈': r' \in ',
            '∉': r' \notin ',
            '⊂': r' \subset ',
            '⊃': r' \supset ',
            '∪': r' \cup ',
            '∩': r' \cap ',
            '∞': r'\infty',
            '∑': r'\sum',
            '∏': r'\prod',
            '∫': r'\int',
            '√': r'\sqrt',
            '∂': r'\partial',
            '∇': r'\nabla',
        }
        # Only convert if inside $ delimiters or if isolated
        for unicode_char, latex in unicode_to_latex.items():
            text = text.replace(unicode_char, latex)
    else:
        # Plain text mode: remove displaystyle wrapper and clean LaTeX
        # First, remove the entire {\\displaystyle ...} blocks (they're duplicates of nearby Unicode)
        text = re.sub(r'\{\\displaystyle[^}]*\}', '', text)
        text = re.sub(r'\\mathbf\s*\{([^}]*)\}', r'\1', text)  # \mathbf{M} -> M
        text = re.sub(r'\\mathit\s*\{([^}]*)\}', r'\1', text)  # \mathit{x} -> x
        text = re.sub(r'\\mathrm\s*\{([^}]*)\}', r'\1', text)  # \mathrm{T} -> T
        text = re.sub(r'\\text\s*\{([^}]*)\}', r'\1', text)    # \text{...} -> ...
        text = re.sub(r'\\times', '×', text)
        text = re.sub(r'\\cdot', '·', text)
        text = re.sub(r'\\Sigma', 'Σ', text)
        text = re.sub(r'\\sigma', 'σ', text)
        text = re.sub(r'\\alpha', 'α', text)
        text = re.sub(r'\\beta', 'β', text)
        text = re.sub(r'\\gamma', 'γ', text)
        text = re.sub(r'\\delta', 'δ', text)
        text = re.sub(r'\\lambda', 'λ', text)
        text = re.sub(r'\\mu', 'μ', text)
        text = re.sub(r'\\pi', 'π', text)
        text = re.sub(r'\\theta', 'θ', text)
        text = re.sub(r'\\phi', 'φ', text)
        text = re.sub(r'\\omega', 'ω', text)
        text = re.sub(r'\\infty', '∞', text)
        text = re.sub(r'\\leq', '≤', text)
        text = re.sub(r'\\geq', '≥', text)
        text = re.sub(r'\\neq', '≠', text)
        text = re.sub(r'\\approx', '≈', text)
        text = re.sub(r'\\in', '∈', text)
        text = re.sub(r'\\subset', '⊂', text)
        text = re.sub(r'\\supset', '⊃', text)
        text = re.sub(r'\\pm', '±', text)
        text = re.sub(r'\^\{?\*\}?', '*', text)  # ^{*} or ^* -> *
        text = re.sub(r'\^\{([^}]*)\}', r'^\1', text)  # ^{2} -> ^2
        text = re.sub(r'_\{([^}]*)\}', r'_\1', text)  # _{ij} -> _ij
        text = re.sub(r'\}', '', text)  # Remove remaining }
        text = re.sub(r',\s*$', '', text, flags=re.MULTILINE)  # Trailing commas
        # Clean up multiple spaces
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Remove excessive blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()


# =============================================================================
# Main Functions
# =============================================================================

def wiki_fetch(query: str, sentences: int = 0) -> str:
    """
    Fetch Wikipedia content and format it for display.
    
    Args:
        query: Search query or exact page title.
        sentences: Number of sentences to return (0 = full summary).
        
    Returns:
        Formatted text content suitable for current environment.
    """
    try:
        # Search for the page
        results = wikipedia.search(query)
        if not results:
            return f"No Wikipedia results found for: {query}"
        
        # Get the page content
        try:
            page = wikipedia.page(results[0], auto_suggest=False)
        except wikipedia.DisambiguationError as e:
            # If disambiguation, try the first option
            page = wikipedia.page(e.options[0], auto_suggest=False)
        
        # Get summary or full content
        if sentences > 0:
            content = wikipedia.summary(page.title, sentences=sentences)
        else:
            content = page.summary
        
        # Format the content
        for_jupyter = is_jupyter()
        formatted = clean_wikipedia_text(content, for_jupyter=for_jupyter)
        
        # Add title header
        if for_jupyter:
            header = f"## {page.title}\n\n"
        else:
            header = f"=== {page.title} ===\n\n"
        
        return header + formatted
    
    except wikipedia.PageError:
        return f"Wikipedia page not found for: {query}"
    except Exception as e:
        return f"Error fetching Wikipedia content: {str(e)}"


def wiki_math(query: str, sentences: int = 0):
    """
    Fetch and display Wikipedia content with proper math rendering.
    
    In Jupyter Notebook: Renders as Markdown with MathJax support.
    In Terminal: Prints clean plain text.
    
    Args:
        query: Search query or exact page title.
        sentences: Number of sentences to return (0 = full summary).
        
    Returns:
        In Jupyter: Rendered Markdown display object.
        In Terminal: None (prints to stdout).
    
    Example:
        >>> wiki_math("Singular value decomposition")
        # Displays formatted content with rendered math
        
        >>> wiki_math("Matrix multiplication", sentences=3)
        # Displays first 3 sentences only
    """
    content = wiki_fetch(query, sentences=sentences)
    
    if is_jupyter():
        from IPython.display import display, Markdown
        return display(Markdown(content))
    else:
        print(content)
        return None


def wiki_content(query: str) -> str:
    """
    Fetch the full Wikipedia page content (not just summary).
    
    Args:
        query: Search query or exact page title.
        
    Returns:
        Full formatted page content.
    """
    try:
        results = wikipedia.search(query)
        if not results:
            return f"No Wikipedia results found for: {query}"
        
        try:
            page = wikipedia.page(results[0], auto_suggest=False)
        except wikipedia.DisambiguationError as e:
            page = wikipedia.page(e.options[0], auto_suggest=False)
        
        for_jupyter = is_jupyter()
        formatted = clean_wikipedia_text(page.content, for_jupyter=for_jupyter)
        
        if for_jupyter:
            header = f"# {page.title}\n\n"
        else:
            header = f"{'='*60}\n{page.title}\n{'='*60}\n\n"
        
        return header + formatted
    
    except Exception as e:
        return f"Error fetching Wikipedia content: {str(e)}"


# =============================================================================
# Demo / Example Usage
# =============================================================================

if __name__ == "__main__":
    
    # Change this to be something else:
    prompt = "When did Gustav Vasa die"
    
    print(f"Fetching {prompt} from Wikipedia...\n")
    
    wiki_math(prompt, sentences=5)


