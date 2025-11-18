#pragma once
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

class Profiler {
  private:
  struct Node
  {
    std::string name;
    Node* parent;
    uint64_t total_ns = 0;
    uint64_t call_count = 0;
    std::map<std::string, std::unique_ptr<Node>> children;
    Node(const std::string& n, Node* p)
      : name(n)
      , parent(p)
    {
    }
  };

  std::unique_ptr<Node> root_;
  std::vector<Node*> node_stack_;
  std::vector<std::chrono::high_resolution_clock::time_point> time_stack_;

  Profiler()
    : root_(std::make_unique<Node>("ROOT", nullptr))
  {
  }
  Profiler(const Profiler&) = delete;
  Profiler& operator=(const Profiler&) = delete;

  // 获取根节点总耗时
  uint64_t getTotalRootNs() const
  {
    uint64_t sum = 0;
    for (auto& kv : root_->children)
      sum += kv.second->total_ns;
    return sum;
  }

  // 打印单个节点及其子节点，子节点按占比排序；同时输出相对父级和相对根的 pct
  void printNode(Node* node, std::ostream& os, int indent, uint64_t parent_ns, uint64_t root_ns) const
  {
    double total_ms = node->total_ns / 1e6;
    double avg_ms = node->call_count ? (total_ms / node->call_count) : 0.0;
    uint64_t child_ns = 0;
    for (auto& kv : node->children)
      child_ns += kv.second->total_ns;
    double self_ms = (node->total_ns > child_ns ? (node->total_ns - child_ns) : 0) / 1e6;
    double pct_parent = parent_ns ? (static_cast<double>(node->total_ns) * 100.0 / parent_ns) : 0.0;
    double pct_root = root_ns ? (static_cast<double>(node->total_ns) * 100.0 / root_ns) : 0.0;

    for (int i = 0; i < indent; ++i)
      os << "├──";
    os << "├──" << node->name << " | total=" << std::fixed << std::setprecision(3) << total_ms << "ms"
       << " | avg=" << std::fixed << std::setprecision(3) << avg_ms
       << "ms"
       // << " | self=" << std::fixed << std::setprecision(3) << self_ms << "ms"
       << " | %parent=" << std::fixed << std::setprecision(2) << pct_parent << "%"
       << " | %root=" << std::fixed << std::setprecision(2) << pct_root << "%"
       << " | calls=" << node->call_count << "\n";

    // 收集子节点并排序
    std::vector<Node*> children;
    children.reserve(node->children.size());
    for (auto& kv : node->children)
      children.push_back(kv.second.get());
    std::sort(children.begin(), children.end(), [](Node* a, Node* b) {
      return a->total_ns > b->total_ns;
    });
    // 递归打印
    for (auto* child : children)
    {
      printNode(child, os, indent + 1, node->total_ns, root_ns);
    }
  }
  // 打印范围内节点：当遇到 endNode 时打印后停止对其子节点的遍历
  void printRange(Node* node, std::ostream& os, int indent, uint64_t parent_ns, uint64_t root_ns, Node* endNode) const
  {
    // 打印当前节点信息
    printNode(node, os, indent, parent_ns, root_ns);
    if (node == endNode)
      return; // 不打印 endNode 的子节点

    // 收集并排序子节点
    std::vector<Node*> children;
    children.reserve(node->children.size());
    for (auto& kv : node->children)
      children.push_back(kv.second.get());
    std::sort(children.begin(), children.end(), [](Node* a, Node* b) {
      return a->total_ns > b->total_ns;
    });
    // 递归打印
    for (auto* child : children)
    {
      printRange(child, os, indent + 1, node->total_ns, root_ns, endNode);
    }
  }

  public:
  // 打印完整报告，包含占比和平均时间，并按占比排序，限制层级深度
  void reportDepth(int maxDepth, std::ostream& os = std::cout) const
  {
    os << "----------------- Profiling Report (MaxDepth=" << maxDepth << ") -----------------\n";
    uint64_t overall_ns = getTotalRootNs();
    // 收集并排序顶级节点
    std::vector<Node*> top_nodes;
    top_nodes.reserve(root_->children.size());
    for (auto& kv : root_->children)
      top_nodes.push_back(kv.second.get());
    std::sort(top_nodes.begin(), top_nodes.end(), [](Node* a, Node* b) {
      return a->total_ns > b->total_ns;
    });
    // 打印，每个顶级节点深度从1开始
    for (auto* child : top_nodes)
    {
      printNodeDepth(child, os, 0, overall_ns, overall_ns, 1, maxDepth);
    }
    os << "----------------------------------------------------\n";
  }

  // 从指定起点打印子树，限制层级深度
  void reportFromDepth(const std::string& startPath, int maxDepth, std::ostream& os = std::cout) const
  {
    if (auto startNode = getNode(startPath))
    {
      os << "----- Subprofile from '" << startPath << "' (MaxDepth=" << maxDepth << ") -----\n";
      uint64_t overall_ns = getTotalRootNs();
      uint64_t parent_ns = (startNode->parent == root_.get()) ? overall_ns : startNode->parent->total_ns;
      // 层级深度从1开始，startNode 作为深度1
      printNodeDepth(startNode, os, 0, parent_ns, startNode->total_ns, 1, maxDepth);
      os << "----- End Subprofile -----\n";
    }
    else
    {
      os << "Start node not found: " << startPath << "\n";
    }
  }

  // 打印从节点 startPath 到 endPath（包含 endPath 本身，但不打印其子节点）的性能报告
  void reportRange(const std::string& startPath, const std::string& endPath, std::ostream& os = std::cout) const
  {
    Node* startNode = getNode(startPath);
    Node* endNode = getNode(endPath);
    if (!startNode)
    {
      os << "Start node not found: " << startPath << "\n";
      return;
    }
    if (!endNode)
    {
      os << "End node not found: " << endPath << "\n";
      return;
    }
    os << "----- Profile Range: " << startPath << " -> " << endPath << " -----\n";
    uint64_t overall_ns = getTotalRootNs();
    uint64_t parent_ns = startNode->parent == root_.get() ? overall_ns : startNode->parent->total_ns;
    uint64_t root_ns = startNode->total_ns;
    // 打印 startNode，并深度遍历，直到 endNode，然后停止其子遍历
    printRange(startNode, os, 0, parent_ns, root_ns, endNode);
    os << "----- End Profile Range -----\n";
  }

  // 获取单例
  static Profiler& instance()
  {
    static Profiler inst;
    return inst;
  }

  // 进入一个新的节点
  void enter(const std::string& name)
  {
    using clk = std::chrono::high_resolution_clock;
    Node* parent = node_stack_.empty() ? root_.get() : node_stack_.back();
    auto& child_ptr = parent->children[name];
    if (!child_ptr)
    {
      child_ptr = std::make_unique<Node>(name, parent);
    }
    Node* child = child_ptr.get();
    node_stack_.push_back(child);
    time_stack_.push_back(clk::now());
  }

  // 退出当前节点
  void exit()
  {
    using clk = std::chrono::high_resolution_clock;
    auto t1 = time_stack_.back();
    time_stack_.pop_back();
    auto t2 = clk::now();
    Node* cur = node_stack_.back();
    node_stack_.pop_back();
    uint64_t ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    cur->total_ns += ns;
    cur->call_count += 1;
  }

  // 清除所有记录
  void clear()
  {
    root_ = std::make_unique<Node>("ROOT", nullptr);
    node_stack_.clear();
    time_stack_.clear();
  }

  // 根据层次路径查找节点，如 "FuncA/FuncB"
  Node* getNode(const std::string& path) const
  {
    Node* node = root_.get();
    size_t pos = 0;
    while (pos < path.size())
    {
      size_t next = path.find('/', pos);
      std::string key = (next == std::string::npos) ? path.substr(pos) : path.substr(pos, next - pos);
      auto it = node->children.find(key);
      if (it == node->children.end())
        return nullptr;
      node = it->second.get();
      if (next == std::string::npos)
        break;
      pos = next + 1;
    }
    return node;
  }

  // 获取某节点的总耗时（毫秒）
  double getTotalMs(const std::string& path) const
  {
    if (auto n = getNode(path))
      return n->total_ns / 1e6;
    return 0.0;
  }
  // 获取某节点的自身耗时（毫秒）
  double getSelfMs(const std::string& path) const
  {
    if (auto n = getNode(path))
    {
      uint64_t child_ns = 0;
      for (auto& kv : n->children)
        child_ns += kv.second->total_ns;
      uint64_t self_ns = n->total_ns > child_ns ? n->total_ns - child_ns : 0;
      return self_ns / 1e6;
    }
    return 0.0;
  }
  // 获取某节点的调用次数
  uint64_t getCallCount(const std::string& path) const
  {
    if (auto n = getNode(path))
      return n->call_count;
    return 0;
  }

  // 打印完整报告，包含占比和平均时间，并按占比排序
  void report(std::ostream& os = std::cout) const
  {
    os << "----------------- Profiling Report -----------------\n";
    // 计算总耗时（根节点所有子节点之和）
    uint64_t overall_ns = getTotalRootNs();
    // 收集子节点并排序
    std::vector<Node*> top_nodes;
    top_nodes.reserve(root_->children.size());
    for (auto& kv : root_->children)
      top_nodes.push_back(kv.second.get());
    std::sort(top_nodes.begin(), top_nodes.end(), [](Node* a, Node* b) {
      return a->total_ns > b->total_ns;
    });
    // 打印：父级和根均为 overall_ns
    for (auto* child : top_nodes)
    {
      printNode(child, os, 0, overall_ns, overall_ns);
    }
    os << "----------------------------------------------------\n";
  }

  // 打印从指定路径开始的子树报告，包含占比和平均时间，并按占比排序
  void reportFrom(const std::string& path, std::ostream& os = std::cout) const
  {
    if (auto root_node = getNode(path))
    {
      os << "----- Subprofile: " << path << " -----\n";
      uint64_t sub_root_ns = root_node->total_ns;
      uint64_t parent_ns = (root_node->parent == this->root_.get()) ? getTotalRootNs() : root_node->parent->total_ns;
      // 对 root_node 自身：parent_ns 用于 pctParent, sub_root_ns 用于 pctRoot
      printNode(root_node, os, 0, parent_ns, sub_root_ns);
      os << "----- End Subprofile -----\n";
    }
    else
    {
      os << "Node not found: " << path << "\n";
    }
  }

  // 限制深度打印节点
  void printNodeDepth(Node* node, std::ostream& os, int indent, uint64_t parent_ns, uint64_t root_ns, int depth, int maxDepth) const
  {
    // 打印当前节点
    double total_ms = node->total_ns / 1e6;
    double avg_ms = node->call_count ? (total_ms / node->call_count) : 0.0;
    uint64_t child_ns = 0;
    for (auto& kv : node->children)
      child_ns += kv.second->total_ns;
    double self_ms = (node->total_ns > child_ns ? node->total_ns - child_ns : 0) / 1e6;
    double pct_parent = parent_ns ? (double(node->total_ns) * 100.0 / parent_ns) : 0.0;
    double pct_root = root_ns ? (double(node->total_ns) * 100.0 / root_ns) : 0.0;

    for (int i = 0; i < indent; ++i)
      os << "├──";
    os << "├──" << node->name << " | total=" << std::fixed << std::setprecision(3) << total_ms << "ms"
       << " | avg=" << std::fixed << std::setprecision(3) << avg_ms
       << "ms"
       // << " | self=" << std::fixed << std::setprecision(3) << self_ms << "ms"
       << " | calls=" << node->call_count << " | %parent=" << std::fixed << std::setprecision(2) << pct_parent << "%"
       << " | %root=" << std::fixed << std::setprecision(2) << pct_root << "%"
       << "\n";

    // 如果已达最大深度，停止遍历
    if (depth >= maxDepth)
      return;

    // 收集并排序子节点
    std::vector<Node*> children;
    children.reserve(node->children.size());
    for (auto& kv : node->children)
      children.push_back(kv.second.get());
    std::sort(children.begin(), children.end(), [](Node* a, Node* b) {
      return a->total_ns > b->total_ns;
    });

    // 递归打印下一层
    for (auto* child : children)
    {
      printNodeDepth(child, os, indent + 1, node->total_ns, root_ns, depth + 1, maxDepth);
    }
  }

  std::string getStackTop() const
  {
    if (node_stack_.empty())
      return "";
    return node_stack_.back()->name;
  }
};

// RAII 作用域包装
class ProfileScope {
  public:
  ProfileScope(const char* name)
  {
    Profiler::instance().enter(name);
  }
  ~ProfileScope()
  {
    Profiler::instance().exit();
  }
};

// 便捷宏
#define PROFILE_SCOPE(name) ProfileScope ANON_VAR(__prof__)(name)
#define PROFILE_FUNCTION() PROFILE_SCOPE(__FUNCTION__)
#define PROFILE_CLEAR() Profiler::instance().clear()
#define PROFILE_GET_TIME(path) Profiler::instance().getTotalMs(path)
#define PROFILE_GET_SELF(path) Profiler::instance().getSelfMs(path)
#define PROFILE_GET_CALLS(path) Profiler::instance().getCallCount(path)
#define PROFILE_REPORT_FROM(path) Profiler::instance().reportFrom(path)
#define PROFILE_REPORT_RANGE(start, end) Profiler::instance().reportRange(start, end)
#define PROFILE_REPORT_DEPTH(depth) Profiler::instance().reportDepth(depth)
#define PROFILE_REPORT_FROM_DEPTH(path, d) Profiler::instance().reportFromDepth(path, d)

// Helper for unique var name in macro
#define ANON_VAR_CAT(a, b) a##b
#define ANON_VAR(a) ANON_VAR_CAT(a, __LINE__)
