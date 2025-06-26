# Contributing to Efficient-NAS-BNN-Pipeline

Thank you for your interest in contributing to this project! This guide will help you get started.

## 🎯 **Ways to Contribute**

- **🐛 Bug Reports**: Found an issue? Please report it!
- **✨ Feature Requests**: Have an idea for improvement? We'd love to hear it!
- **📝 Documentation**: Help improve our documentation
- **🔧 Code Contributions**: Submit bug fixes or new features
- **📊 Dataset Adaptations**: Add support for new datasets
- **🏗️ Architecture Extensions**: Contribute new model architectures

## 🚀 **Getting Started**

1. **Fork the repository**
2. **Clone your fork**:
   ```bash
   git clone https://github.com/YourUsername/Efficient-NAS-BNN-Pipeline.git
   cd Efficient-NAS-BNN-Pipeline
   ```
3. **Set up the development environment**:
   ```bash
   python -m venv nasbnn
   source nasbnn/bin/activate  # Linux/macOS
   # or
   .\nasbnn\Scripts\activate   # Windows
   pip install -r requirements.txt
   ```

## 📋 **Development Guidelines**

### **Code Style**
- Follow PEP 8 for Python code
- Use meaningful variable names
- Add docstrings for functions and classes
- Keep functions focused and small

### **Testing**
- Test your changes with different datasets if applicable
- Ensure the Jupyter notebook runs without errors
- Verify that resume functionality works correctly

### **Documentation**
- Update README.md if adding new features
- Add inline comments for complex logic
- Update the Jupyter notebook if workflow changes

## 🔄 **Dataset Adaptation Process**

When adding support for a new dataset:

1. **Create data preparation script** (`prepare_[dataset_name].py`)
2. **Add model architecture** in `models/superbnn.py` if needed
3. **Update default parameters** in scripts
4. **Test the complete pipeline**
5. **Update documentation**

## 📝 **Pull Request Process**

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following the guidelines above

3. **Test thoroughly**:
   - Run the pipeline with your changes
   - Check that existing functionality still works
   - Test on different configurations if possible

4. **Commit your changes**:
   ```bash
   git add .
   git commit -m "feat: add support for [feature description]"
   ```

5. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Create a Pull Request** with:
   - Clear description of changes
   - Screenshots/logs if applicable
   - Reference to any related issues

## 🐛 **Reporting Issues**

When reporting bugs, please include:

- **Operating System** and version
- **Python version** and virtual environment details
- **PyTorch version** and CUDA version
- **Complete error message** and stack trace
- **Steps to reproduce** the issue
- **Expected vs actual behavior**

## 💡 **Feature Requests**

For feature requests, please describe:

- **Use case**: What problem does this solve?
- **Proposed solution**: How should it work?
- **Alternatives considered**: Other approaches you've thought of
- **Additional context**: Any other relevant information

## 🏷️ **Commit Message Guidelines**

Use conventional commits format:

- `feat:` for new features
- `fix:` for bug fixes
- `docs:` for documentation changes
- `style:` for formatting changes
- `refactor:` for code refactoring
- `test:` for adding tests
- `chore:` for maintenance tasks

Examples:
```
feat: add support for ResNet architectures
fix: resolve CUDA memory issue in training
docs: update installation instructions
```

## 📞 **Getting Help**

- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Code Review**: We provide constructive feedback on PRs

## 🎉 **Recognition**

Contributors will be acknowledged in the project documentation and releases. We appreciate all forms of contribution, no matter how small!

---

**Thank you for helping make this project better! 🚀**
