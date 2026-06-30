import { makeAutoObservable } from 'mobx'
import { SERVER_URL } from '../utils'

class GlobalUser {
  userInfo: Partial<User.UserEntity> = {}
  selectedDatabase: string = ''
  availableDatabases: Array<{ name: string; description: string }> = []

  constructor() {
    makeAutoObservable(this)
  }

  async getUserDetail() {
    // const res = await getCurrentUserInfo()
    // this.userInfo = res?.data
    // new WebSee(res?.data?.username)
    this.userInfo = {
      roles: [
        {
          id: 5,
          name: '超级管理员',
          description: '拥有所有查看和操作功能',
          adminCount: 0,
          status: 1,
          sort: 5
        }
      ],
      icon: 'http://jinpika-1308276765.cos.ap-shanghai.myqcloud.com/bootdemo-file/20221220/src=http___desk-fd.zol-img.com.cn_t_s960x600c5_g2_M00_00_0B_ChMlWl6yKqyILFoCACn-5rom2uIAAO4DgEODxAAKf7-298.jpg&refer=http___desk-fd.zol-img.com.png',
      username: 'admin'
    }
  }

  setUserInfo(user: Partial<User.UserEntity>) {
    this.userInfo = user
  }

  // 设置当前选中的数据库
  setSelectedDatabase(database: string) {
    this.selectedDatabase = database
    // 保存到localStorage
    localStorage.setItem('selectedDatabase', database)
  }

  // 设置可用数据库列表
  setAvailableDatabases(databases: Array<{ name: string; description: string }>) {
    this.availableDatabases = databases
    // 如果还没有选中数据库且有可用数据库，选择第一个
    if (!this.selectedDatabase && databases.length > 0) {
      this.setSelectedDatabase(databases[0].name)
      return
    }

    // 如果当前选中的数据库不在可用列表中，则回退到第一个可用数据库
    if (this.selectedDatabase) {
      const existsInAvailable = databases.some(db => db.name === this.selectedDatabase)
      if (!existsInAvailable && databases.length > 0) {
        this.setSelectedDatabase(databases[0].name)
      }
    }
  }

  // 从localStorage恢复选中的数据库
  restoreSelectedDatabase() {
    const saved = localStorage.getItem('selectedDatabase')
    if (saved) {
      // 如果已加载可用数据库，则进行校验；否则先恢复，待可用数据库加载后再由 setAvailableDatabases 校验
      if (this.availableDatabases.length > 0) {
        const existsInAvailable = this.availableDatabases.some(db => db.name === saved)
        if (existsInAvailable) {
          this.selectedDatabase = saved
        } else if (this.availableDatabases.length > 0) {
          this.setSelectedDatabase(this.availableDatabases[0].name)
        }
      } else {
        this.selectedDatabase = saved
      }
    }
  }

  // 获取数据库列表
  async loadDatabases() {
    try {
      const response = await fetch(`${SERVER_URL}/databases`)
      if (response.ok) {
        const databases = await response.json()
        this.setAvailableDatabases(databases)
        return databases
      }
    } catch (error) {
      console.error('加载数据库列表失败:', error)
    }
    return []
  }
}

export const storeGlobalUser = new GlobalUser()
