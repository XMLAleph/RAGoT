"""
# 需要1.实现两个页面（首页和结果页），2.修改代码，把从知识库检索的环节换成web（顺便整理代码，合并Sc-IRCoT和RAGoT的代码）
# 页面优先确保实现功能，再调整样式
# 就使用flask实现，优先保证流程完整

# 已完成：
# 1 实现完整的首页功能
    1.1 通过下拉菜单配置系统设置（一个按钮提交多个表单，重要）（另外按钮位置等也想一下怎么调整）

    1.2 点击问题样例直接搜索相应内容
        目前还无法实现这个功能（能实现每次打开页面时传入的值都是随机的，但点击相应的按钮没法把值填入搜索框）
        这个功能最简单的实现应该可以直接用超链接（构造查询的url），但最好还是实现“填入文本框”的形式
        现在的情况是点击按钮没有任何反应，但按理说这种做法是能把值直接填入<input>的
        另外除了onclick，应该还可以考虑onmousehover，但问题在于没有反应
        
# 2 实现web接入（在后端测试）
    测试阶段先不考虑RAGoT的代码，先用IRCoT和Sc-IRCoT的代码，调整样式后再考虑代码的合并和整理（应该没有本质的区别）
    另外调用检索问答框架的方式（涉及到config载入）也需要考虑
    最优先考虑web接入，后面完善结果页的时候用本地知识库（尽量降低调用web的次数）
   
# 3 结果页功能
    3.1 搜索结果分页展示（重要）
      分页展示和现在的流程逻辑合并，采用.json文件进行管理
      另外分页的样式还需要调整（目前虽然应该去除了边框，但背景颜色是白色，和页面背景不一致，需要调整
      
    3.2 问题答案等展示（超链接）
        问题答案展示需要动态生成html，动态生成html可以参考flask paginate的做法
        看下直接构建url的做法是否可行，可能需要另一个app方法
        单独的url(标题->文档)
            问题在于本地知识库和web的处理方式不一样（希望是都能返回一个url）
            如果是本地知识库，url就是title 但点击url怎么跳转？或者url怎么构建？
            
    3.3 搜索框
        先把index.html的搜索框直接移到result.html（调整位置）
            这里需要考虑实现无动画效果的搜索框，按之前的想法看下情况，主要是位置布局   
            现在搜索框按钮的位置有问题，后面需要调整一下元素布局    

# 需要：
# 4 调整整体样式
    确保流程完整之后，根据设计图完善各个页面的元素，从初始页面开始逐个页面检查
        先用ircot/sc-ircot测试流程（除hotpotqa外每种检索源分别测试一次）
        测试每种情况下的完整功能，包括翻页，详情页面等
        将url引入问答系统代码：1.调整dataset_reader，初始化data["urls"]；2.framework.py调整返回结果，加入urls；3.scircot.py中在state.data["titles"]和["paras"]变化的位置（ircot和sc-ircot就是retriever，ragot还有voter，需要另外注意）加上state.data["urls"]
    另外后面考虑下有没有可能实现并行请求api
    hover效果持续(点击一下展开，之后也不会关闭)
    
    search-box height: 45px;
    search-txt line-height: 27px;padding-bottom: 10px;
"""