# Claude Code Project Guidelines

## Development Modes

### PROTOTYPE Mode (Default)
**Trigger**: When user says "prototype", "POC", "proof of concept", or doesn't specify mode
**Philosophy**: Fast iteration, clean experimentation, demonstrate core value

**Attributes:**
1. **Structure**: Simple, clear organization - whatever makes sense for the project type
2. **Documentation**: 
   - Brief README with implementation approach
   - Clear usage instructions for both developers and non-technical users
   - "Steve Jobs prototype stage" clarity - intuitive and compelling
3. **Code Quality**: Clean, functional, demonstrates core aspects without bloat
4. **Technology Stack**: Modern, actively maintained tools - use what works best
5. **Version Control**: Always initialize git with appropriate .gitignore

**Documentation Standards:**
- README must have: What it does, Why it matters, How to use it
- Include both developer setup AND end-user instructions
- Maximum 200 words for overview, bullet points for steps
- Include a "Quick Start" section that works in < 5 minutes

### PRODUCTION Mode
**Trigger**: When user explicitly says "production", "production-ready", or "mature this prototype"
**Philosophy**: Scalable, maintainable, ready for real-world use

**Enhanced Attributes:**
1. **Structure**: Well-organized, modular architecture appropriate for project type
2. **Documentation**: Comprehensive docs, troubleshooting guides, examples
3. **Quality**: Proper error handling, edge case coverage, robustness
4. **Testing**: Appropriate test coverage for the project type
5. **Deployment**: Ready for distribution/deployment in target environment

## Project Initiation

**For Prototype Mode:**
- "Create a prototype for..."
- "Build a POC that..."
- "I need a quick implementation of..."

**For Production Mode:**
- "Make this production-ready"
- "Convert to production grade"
- "I need a robust solution for..."

## Guidelines for Claude Code Agent

1. **Always ask for mode clarification if ambiguous**
2. **Default to PROTOTYPE mode unless explicitly specified**
3. **For prototypes**: Focus on core functionality, keep it simple and clean
4. **For production**: Add comprehensive error handling, testing, and documentation
5. **Always initialize git repository with appropriate .gitignore**
6. **Use modern, actively maintained dependencies only**
7. **Create clear README with usage instructions for both technical and non-technical users**
8. **Adapt structure and approach to the specific project type and technology**
9. **if python packages need to be installed, create virtual environment named venv and install there**

## Success Metrics

**Prototype Success**: Can demo core functionality in < 10 minutes to non-technical stakeholder
**Production Success**: Robust, well-documented, ready for handoff or deployment
